import heapq
import math

from config import (
    Node, Constraint, MAP_SIZE, R_MIN, STEP_SIZE, THETA_MAX, D_SAFE,
    GOAL_THRESHOLD, LOW_LEVEL_MAX_STEPS,
)


# ========================= 全局计数器 =========================
_find_path_call_count = 0


def get_find_path_call_count():
    """返回 find_path 被调用的总次数。"""
    return _find_path_call_count


def reset_find_path_call_count():
    """重置 find_path 调用计数器。"""
    global _find_path_call_count
    _find_path_call_count = 0


def _euclidean_distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _euclidean_distance_tuple(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _violates_constraint(node, constraints, time, agent_id):
    """检查节点是否违反任何时间-空间约束（点约束）。

    论文中的空间约束是点约束：禁止在时刻 t 到达某个具体坐标。
    约束半径 = STEP_SIZE/2 = 1.0，比 D_SAFE=3 的圆柱体约束宽松得多，
    但仍能有效阻止 agent 通过冲突点，绕行代价较小。
    """
    for constraint in constraints:
        if constraint.agent_id != agent_id:
            continue
        if constraint.time != time:
            continue
        px, py, pz = constraint.pos
        # 3D 欧氏距离 < epsilon → 点约束
        CONSTRAINT_EPSILON = 1.0  # STEP_SIZE / 2，一个「半步」的范围
        dist = math.sqrt((node.x - px)**2 + (node.y - py)**2 + (node.z - pz)**2)
        if dist < CONSTRAINT_EPSILON:
            return True
    return False


def get_successors(cur_node, par_node, goal):
    """Generate valid successor nodes from the current node following Dubins curve,
    plus a special action heading directly toward the goal.
    """
    # 修复1：初始节点时使用 cur_node.yaw，而不是硬编码为 0
    if par_node is None:
        yaw = cur_node.yaw  # 保留 start_node 中预置的朝目标方向的初始航向
    else:
        yaw = math.atan2(cur_node.y - par_node.y, cur_node.x - par_node.x)

    dpsi = STEP_SIZE / R_MIN
    c = R_MIN * math.sqrt(2 * (1 - math.cos(dpsi)))
    max_dz = STEP_SIZE * math.tan(math.radians(THETA_MAX))
    successors = []

    # 修复2：解耦转向方向与高度变化 + 增加大角度转向
    # 每个转向等级配三种高度选择（爬升/平飞/俯冲）
    turn_offsets = [
        1.0 * dpsi,   # 大左转（约 76°，STEP_SIZE=2 时）
        0.5 * dpsi,   # 中左转
        0.0,          # 直行
        -0.5 * dpsi,  # 中右转
        -1.0 * dpsi,  # 大右转
    ]
    dz_options = [max_dz, 0.0, -max_dz]  # 爬升 / 平飞 / 俯冲（全量）

    for offset in turn_offsets:
        for dz in dz_options:
            if offset == 0.0:
                # 直行：沿当前航向
                x_new = cur_node.x + STEP_SIZE * math.cos(yaw)
                y_new = cur_node.y + STEP_SIZE * math.sin(yaw)
            else:
                # 转弯：圆弧运动
                mid_angle = yaw + offset
                x_new = cur_node.x + c * math.cos(mid_angle)
                y_new = cur_node.y + c * math.sin(mid_angle)
            z_new = cur_node.z + dz

            # 爬升/俯冲角约束（水平距离）
            horizontal_dist = math.sqrt((x_new - cur_node.x)**2 + (y_new - cur_node.y)**2)
            if horizontal_dist > 1e-9:
                slope = abs(z_new - cur_node.z) / horizontal_dist
                if slope > math.tan(math.radians(THETA_MAX)) + 1e-9:
                    continue
            # 额外保险检查
            if abs(z_new - cur_node.z) > max_dz + 1e-9:
                continue

            # 边界约束
            if not (0 <= x_new <= MAP_SIZE[0] and 0 <= y_new <= MAP_SIZE[1] and 0 <= z_new <= MAP_SIZE[2]):
                continue

            # 生成节点
            new_node = Node(x_new, y_new, z_new, parent=cur_node)
            new_node.g = cur_node.g + _euclidean_distance(cur_node, new_node)
            new_node.h = math.sqrt((goal.x - x_new)**2 + (goal.y - y_new)**2 + (goal.z - z_new)**2)
            # H_WEIGHT=1.1：略高于标准A*，加速收敛但不至于过于贪婪
            H_WEIGHT = 1.1
            new_node.f = new_node.g + H_WEIGHT * new_node.h
            new_node.yaw = yaw + offset
            successors.append(new_node)

    # ==================== 直接指向目标点的引导动作 ====================
    target_yaw_raw = math.atan2(goal.y - cur_node.y, goal.x - cur_node.x)
    # 修复8：引导动作航向必须受最大偏转角限制，不能瞬间突变
    yaw_diff = target_yaw_raw - yaw
    # 将角度差归一化到 [-π, π]
    while yaw_diff > math.pi:
        yaw_diff -= 2 * math.pi
    while yaw_diff < -math.pi:
        yaw_diff += 2 * math.pi
    max_turn = 1.5 * dpsi  # 与最大动作偏转角一致
    clamped_diff = max(-max_turn, min(max_turn, yaw_diff))
    target_yaw = yaw + clamped_diff  # 不能超过最大偏转角

    x_goal = cur_node.x + STEP_SIZE * math.cos(target_yaw)
    y_goal = cur_node.y + STEP_SIZE * math.sin(target_yaw)
    # 引导动作高度也向目标趋近
    dz_raw = goal.z - cur_node.z
    dz_clamped = max(-max_dz, min(max_dz, dz_raw))
    z_goal = cur_node.z + dz_clamped

    # 爬升角检查（引导动作）
    horizontal_dist = STEP_SIZE
    slope = abs(z_goal - cur_node.z) / horizontal_dist if horizontal_dist > 0 else 0
    if slope <= math.tan(math.radians(THETA_MAX)) + 1e-9:
        if 0 <= x_goal <= MAP_SIZE[0] and 0 <= y_goal <= MAP_SIZE[1] and 0 <= z_goal <= MAP_SIZE[2]:
            new_node = Node(x_goal, y_goal, z_goal, parent=cur_node)
            new_node.g = cur_node.g + _euclidean_distance(cur_node, new_node)
            new_node.h = math.sqrt((goal.x - x_goal)**2 + (goal.y - y_goal)**2 + (goal.z - z_goal)**2)
            # 标准 A*：无转弯惩罚
            H_WEIGHT = 1.1
            new_node.f = new_node.g + H_WEIGHT * new_node.h
            new_node.yaw = target_yaw
            successors.append(new_node)

    return successors


def find_path(start, goal, constraints, agent_id):
    """Find a sparse A* path from start to goal with time-space constraints.

    Returns:
        list of (x, y, z, time) tuples, or None if no path found.
    """
    global _find_path_call_count
    _find_path_call_count += 1  # 每次调用递增

    # 计算起点指向目标的方向角（弧度）
    print(f"[底层] 为 agent {agent_id} 规划路径，约束数={len(constraints)}")
    start_yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
    H_WEIGHT = 1.1  # 与 get_successors 保持一致
    start_node = Node(start[0], start[1], start[2], parent=None, g=0.0,
                      h=_euclidean_distance_tuple(start, goal), yaw=start_yaw)
    start_node.f = start_node.g + H_WEIGHT * start_node.h

    open_set = []
    heapq.heappush(open_set, (start_node.f, 0, start_node, 0))
    closed_set = set()
    entry_count = 1
    goal_node = Node(goal[0], goal[1], goal[2])

    count = 0
    MAX_STEPS = LOW_LEVEL_MAX_STEPS
    PRINT_INTERVAL = 2000

    while open_set:
        count += 1
        if count > MAX_STEPS:
            print(f"[底层] agent {agent_id} 搜索超时 (>{count - 1}步)")
            return None
        _, _, current, current_time = heapq.heappop(open_set)
        if count % PRINT_INTERVAL == 0:
            dist = _euclidean_distance_tuple((current.x, current.y, current.z), goal)
            print(f"[底层] agent {agent_id} 步数 {count}, 位置 ({current.x:.1f},{current.y:.1f},{current.z:.1f}), "
                  f"距终点 {dist:.1f}, yaw={math.degrees(current.yaw):.0f}°")

        # 修复4：终点判定 — 欧氏距离 < GOAL_THRESHOLD 即认为到达
        dist_to_goal = _euclidean_distance_tuple((current.x, current.y, current.z), goal)
        if dist_to_goal < GOAL_THRESHOLD:
            # ===== 修复：正确构建时间递增的路径 =====
            # 回溯链：current (t=current_time) -> parent (t-1) -> ... -> start (t=0)
            # 构建 [start(t=0), (t=1), ..., current(t=current_time), goal(t=current_time+1)]
            path_rev = []
            node = current
            t = current_time
            while node is not None:
                path_rev.append((node.x, node.y, node.z, t))
                node = node.parent
                t -= 1
            # path_rev = [current_t, t-1, ..., start_t=0]
            # 反转后 path = [start(t=0), ..., current(t=current_time)]
            path_rev.reverse()
            # 追加 goal
            path_rev.append((goal[0], goal[1], goal[2], current_time + 1))
            print(f"[底层] agent {agent_id} 成功找到路径！共 {count} 步，路径长度 {len(path_rev)}")
            return path_rev

        # 修复5：closed_set 包含航向（量化到约 5.7° 即 0.1 rad）
        YAW_QUANT = 0.4  # 弧度，约 23 度（加粗以加速搜索）
        closed_key = (
            round(current.x, 1),
            round(current.y, 1),
            round(current.z, 1),
            round(current.yaw / YAW_QUANT)
        )

        if closed_key in closed_set:
            continue
        closed_set.add(closed_key)

        successors = get_successors(current, current.parent, goal_node)
        for successor in successors:
            successor_time = current_time + 1
            if _violates_constraint(successor, constraints, successor_time, agent_id):
                continue

            successor_key = (
                round(successor.x, 1),
                round(successor.y, 1),
                round(successor.z, 1),
                round(successor.yaw / YAW_QUANT)
            )
            if successor_key in closed_set:
                continue

            successor.parent = current
            heapq.heappush(open_set, (successor.f, entry_count, successor, successor_time))
            entry_count += 1

    print(f"[底层] agent {agent_id} Open Set 耗尽，未找到路径")
    return None
