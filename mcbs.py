import copy
import heapq
import math
import time

from conflict import detect_conflicts
from sparse_astar import find_path, reset_find_path_call_count, get_find_path_call_count
from config import (
    Constraint, OMEGA_S, OMEGA_T, D_SAFE, V,
    TIMEOUT, MAX_CT_NODES,
)


def _path_euclidean_length(path):
    """计算单条路径的欧氏距离总长度（相邻点间欧氏距离之和）。"""
    if not path or len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        p1 = path[i][:3]
        p2 = path[i + 1][:3]
        total += math.dist(p1, p2)
    return total


def _solution_metrics(solution):
    """从解中提取所有评估指标。

    Returns:
        dict with keys:
          - per_uav_path_length: list of float
          - total_path_length: float
          - min_pairwise_dist: float (all timesteps, all pairs)
          - per_uav_arrival_time: list of float (path_len / V)
          - arrival_time_std: float
    """
    n = len(solution)
    per_uav_len = [_path_euclidean_length(p) for p in solution]
    total_len = sum(per_uav_len)
    per_uav_arrival = [len(p) / V for p in solution]  # V=1 时即路径步数

    # 到达时间标准差
    if n > 1:
        mean_arrival = sum(per_uav_arrival) / n
        arrival_std = math.sqrt(
            sum((t - mean_arrival) ** 2 for t in per_uav_arrival) / n
        )
    else:
        arrival_std = 0.0

    # 所有时刻两两之间最小欧氏距离
    max_len = max(len(p) for p in solution)
    min_dist = float('inf')
    for t in range(max_len):
        active_positions = []
        for i, p in enumerate(solution):
            if t < len(p):
                active_positions.append((i, p[t][:3]))
        for a in range(len(active_positions)):
            for b in range(a + 1, len(active_positions)):
                dist = math.dist(active_positions[a][1], active_positions[b][1])
                if dist < min_dist:
                    min_dist = dist
    if min_dist == float('inf'):
        min_dist = 0.0

    return {
        'per_uav_path_length': per_uav_len,
        'total_path_length': total_len,
        'min_pairwise_dist': min_dist,
        'per_uav_arrival_time': per_uav_arrival,
        'arrival_time_std': arrival_std,
    }


def count_space_conflicts(solution):
    """统计解中空间冲突数量（用于 cost 计算）。"""
    conflict = detect_conflicts(solution)
    if conflict is not None and conflict.get('type') == 'vertex':
        return 1
    return 0


def compute_node_cost(solution):
    """计算约束树节点的 cost（用于优先队列排序）。"""
    base_cost = sum(len(path) for path in solution)
    n_s = count_space_conflicts(solution)
    max_len = max(len(p) for p in solution)
    time_diff_sum = sum(max_len - len(p) for p in solution)
    return base_cost + OMEGA_S * n_s + OMEGA_T * time_diff_sum


class CTNode:
    """Constraint Tree node for MCBS high-level search."""

    __slots__ = ('solution', 'cost', 'constraints')

    def __init__(self, solution, cost, constraints=None):
        self.solution = solution
        self.cost = cost
        self.constraints = constraints or []

    def __lt__(self, other):
        if self.cost != other.cost:
            return self.cost < other.cost
        return id(self) < id(other)


def mcbs(start_list, goal_list):
    """High-level MCBS search for conflict-free multi-UAV paths.

    Returns:
        (solution, stats_dict) where stats_dict contains all experimental metrics,
        or (None, stats_dict) on failure/timeout.
    """
    stats = {
        'success': False,
        'total_time_s': 0.0,
        'ct_nodes': 0,
        'find_path_calls': 0,
        'per_uav_path_length': [],
        'total_path_length': 0.0,
        'min_pairwise_dist': 0.0,
        'per_uav_arrival_time': [],
        'arrival_time_std': 0.0,
        'notes': '',
    }

    if len(start_list) != len(goal_list):
        stats['notes'] = 'start/goal length mismatch'
        return None, stats

    reset_find_path_call_count()
    t_start = time.perf_counter()

    root_solution = []
    num_agents = len(start_list)

    for agent_id in range(num_agents):
        path = find_path(start_list[agent_id], goal_list[agent_id], [], agent_id)
        if path is None:
            elapsed = time.perf_counter() - t_start
            stats['total_time_s'] = round(elapsed, 4)
            stats['ct_nodes'] = 0
            stats['find_path_calls'] = get_find_path_call_count()
            stats['notes'] = f'no initial path for agent {agent_id}'
            return None, stats
        root_solution.append(path)

    root_cost = compute_node_cost(root_solution)
    root = CTNode(root_solution, root_cost, constraints=[])
    open_queue = []
    heapq.heappush(open_queue, root)

    ct_nodes = 0  # open_queue 弹出次数

    while open_queue:
        # --- 超时检查 ---
        elapsed = time.perf_counter() - t_start
        if elapsed > TIMEOUT:
            stats['total_time_s'] = round(elapsed, 4)
            stats['ct_nodes'] = ct_nodes
            stats['find_path_calls'] = get_find_path_call_count()
            stats['notes'] = f'timeout after {TIMEOUT}s'
            return None, stats

        # --- 节点数上限检查 ---
        if ct_nodes >= MAX_CT_NODES:
            elapsed = time.perf_counter() - t_start
            stats['total_time_s'] = round(elapsed, 4)
            stats['ct_nodes'] = ct_nodes
            stats['find_path_calls'] = get_find_path_call_count()
            stats['notes'] = f'exceeded max CT nodes ({MAX_CT_NODES})'
            return None, stats

        node = heapq.heappop(open_queue)
        ct_nodes += 1

        conflict = detect_conflicts(node.solution)

        print(f"[MCBS] CT节点 #{ct_nodes}, cost={node.cost}, "
              f"约束数={len(node.constraints)}, 冲突={conflict}")

        if conflict is None:
            # ===== 找到无冲突解 =====
            elapsed = time.perf_counter() - t_start
            metrics = _solution_metrics(node.solution)
            stats.update({
                'success': True,
                'total_time_s': round(elapsed, 4),
                'ct_nodes': ct_nodes,
                'find_path_calls': get_find_path_call_count(),
                'per_uav_path_length': metrics['per_uav_path_length'],
                'total_path_length': metrics['total_path_length'],
                'min_pairwise_dist': metrics['min_pairwise_dist'],
                'per_uav_arrival_time': metrics['per_uav_arrival_time'],
                'arrival_time_std': metrics['arrival_time_std'],
                'notes': 'success',
            })
            print(f"[MCBS] 找到无冲突解！耗时 {elapsed:.3f}s, CT节点 {ct_nodes}, "
                  f"低层调用 {stats['find_path_calls']}次")
            return node.solution, stats

        # --- 根据冲突类型分支 ---
        if conflict['type'] == 'vertex':
            agent1 = conflict['agent1']
            agent2 = conflict['agent2']
            time_t = conflict['time']
            pos1 = conflict['pos1']
            pos2 = conflict['pos2']

            # 圆柱体约束：只约束 XY，不区分 Z。
            # 配合 _violates_constraint 的 XY-only 检查，一次约束就能
            # 迫使 agent 做水平绕行，避免对称爬升级联。

            # 分支1：约束 agent1
            new_c1 = copy.deepcopy(node.constraints)
            new_c1.append(Constraint(agent1, pos1, time_t))
            new_sol1 = copy.deepcopy(node.solution)
            new_path1 = find_path(start_list[agent1], goal_list[agent1],
                                  new_c1, agent1)
            if new_path1 is not None:
                new_sol1[agent1] = new_path1
                heapq.heappush(open_queue,
                    CTNode(new_sol1, compute_node_cost(new_sol1), new_c1))

            # 分支2：约束 agent2
            new_c2 = copy.deepcopy(node.constraints)
            new_c2.append(Constraint(agent2, pos2, time_t))
            new_sol2 = copy.deepcopy(node.solution)
            new_path2 = find_path(start_list[agent2], goal_list[agent2],
                                  new_c2, agent2)
            if new_path2 is not None:
                new_sol2[agent2] = new_path2
                heapq.heappush(open_queue,
                    CTNode(new_sol2, compute_node_cost(new_sol2), new_c2))

        elif conflict['type'] == 'edge':
            # 边冲突：两机在 [t, t+1] 之间对穿
            agent1 = conflict['agent1']
            agent2 = conflict['agent2']
            time_t = conflict['time']
            pos_i_t = conflict['pos_i_t']
            pos_i_t1 = conflict['pos_i_t1']
            pos_j_t = conflict['pos_j_t']
            pos_j_t1 = conflict['pos_j_t1']

            # 分支1：约束 agent1 在 time_t 不能去 pos_i_t，在 time_t+1 不能去 pos_i_t1
            new_c1 = copy.deepcopy(node.constraints)
            new_c1.append(Constraint(agent1, pos_i_t, time_t))
            new_c1.append(Constraint(agent1, pos_i_t1, time_t + 1))
            new_sol1 = copy.deepcopy(node.solution)
            new_path1 = find_path(start_list[agent1], goal_list[agent1],
                                  new_c1, agent1)
            if new_path1 is not None:
                new_sol1[agent1] = new_path1
                heapq.heappush(open_queue,
                    CTNode(new_sol1, compute_node_cost(new_sol1), new_c1))

            # 分支2：约束 agent2 在 time_t 不能去 pos_j_t，在 time_t+1 不能去 pos_j_t1
            new_c2 = copy.deepcopy(node.constraints)
            new_c2.append(Constraint(agent2, pos_j_t, time_t))
            new_c2.append(Constraint(agent2, pos_j_t1, time_t + 1))
            new_sol2 = copy.deepcopy(node.solution)
            new_path2 = find_path(start_list[agent2], goal_list[agent2],
                                  new_c2, agent2)
            if new_path2 is not None:
                new_sol2[agent2] = new_path2
                heapq.heappush(open_queue,
                    CTNode(new_sol2, compute_node_cost(new_sol2), new_c2))

        elif conflict['type'] == 'time':
            agent = conflict['agent']
            path = node.solution[agent]
            # 时间冲突：短路径 UAV 需要延长路径以实现时间协同。
            # 策略1（goal-waiting）：直接在终点等待，扩展 1 步 —— O(1) 分支。
            # 策略2（论文约束分支）：约束路径末段的少量中间点，
            #   迫使低层 A* 绕行以自然延长路径 —— 上限 3 个分支。
            # 两者结合，兼顾效率与完备性。

            # 策略1：goal-waiting —— 终点等待一步
            new_sol_wait = copy.deepcopy(node.solution)
            wait_path = new_sol_wait[agent]
            goal_pos = wait_path[-1][:3]
            last_t = wait_path[-1][3]
            wait_path.append((goal_pos[0], goal_pos[1], goal_pos[2], last_t + 1))
            new_sol_wait[agent] = wait_path
            new_cost_wait = compute_node_cost(new_sol_wait)
            heapq.heappush(open_queue,
                CTNode(new_sol_wait, new_cost_wait, copy.deepcopy(node.constraints)))

            # 策略2：约束末段中间点（最多 3 个），促使绕行延长
            L = len(path)
            MAX_TIME_BRANCHES = 3
            indices = list(range(max(1, L - 6), L - 1))  # 只取最后 5 个中间点
            if len(indices) > MAX_TIME_BRANCHES:
                step = len(indices) / MAX_TIME_BRANCHES
                indices = [indices[int(i * step)] for i in range(MAX_TIME_BRANCHES)]
            for i in indices:
                x, y, z, _ = path[i]
                new_constraints = copy.deepcopy(node.constraints)
                new_constraints.append(Constraint(agent, (x, y, z), i))
                new_solution = copy.deepcopy(node.solution)
                new_path = find_path(start_list[agent], goal_list[agent],
                                    new_constraints, agent)
                if new_path is None:
                    continue
                new_solution[agent] = new_path
                new_cost = compute_node_cost(new_solution)
                heapq.heappush(open_queue,
                    CTNode(new_solution, new_cost, new_constraints))

    # open_queue 耗尽
    elapsed = time.perf_counter() - t_start
    stats['total_time_s'] = round(elapsed, 4)
    stats['ct_nodes'] = ct_nodes
    stats['find_path_calls'] = get_find_path_call_count()
    stats['notes'] = 'open queue exhausted'
    return None, stats


def cbs(start_list, goal_list):
    """CBS (Conflict-Based Search) —— 仅处理空间冲突的基线算法。

    与 MCBS 的区别：
    - 不检测时间冲突，只处理 vertex 和 edge 冲突
    - cost = sum(len(path))，不加 OMEGA_S / OMEGA_T 惩罚
    - 用于对比验证 MCBS 时间协同的有效性

    Returns:
        (solution, stats_dict)
    """
    stats = {
        'success': False,
        'total_time_s': 0.0,
        'ct_nodes': 0,
        'find_path_calls': 0,
        'per_uav_path_length': [],
        'total_path_length': 0.0,
        'min_pairwise_dist': 0.0,
        'per_uav_arrival_time': [],
        'arrival_time_std': 0.0,
        'notes': '',
    }

    if len(start_list) != len(goal_list):
        stats['notes'] = 'start/goal length mismatch'
        return None, stats

    reset_find_path_call_count()
    t_start = time.perf_counter()

    root_solution = []
    num_agents = len(start_list)

    for agent_id in range(num_agents):
        path = find_path(start_list[agent_id], goal_list[agent_id], [], agent_id)
        if path is None:
            elapsed = time.perf_counter() - t_start
            stats['total_time_s'] = round(elapsed, 4)
            stats['ct_nodes'] = 0
            stats['find_path_calls'] = get_find_path_call_count()
            stats['notes'] = f'CBS: no initial path for agent {agent_id}'
            return None, stats
        root_solution.append(path)

    # CBS cost = sum of path lengths (no conflict penalty terms)
    root_cost = sum(len(p) for p in root_solution)
    root = CTNode(root_solution, root_cost, constraints=[])
    open_queue = []
    heapq.heappush(open_queue, root)

    ct_nodes = 0

    while open_queue:
        elapsed = time.perf_counter() - t_start
        if elapsed > TIMEOUT:
            stats['total_time_s'] = round(elapsed, 4)
            stats['ct_nodes'] = ct_nodes
            stats['find_path_calls'] = get_find_path_call_count()
            stats['notes'] = f'CBS timeout after {TIMEOUT}s'
            return None, stats

        if ct_nodes >= MAX_CT_NODES:
            elapsed = time.perf_counter() - t_start
            stats['total_time_s'] = round(elapsed, 4)
            stats['ct_nodes'] = ct_nodes
            stats['find_path_calls'] = get_find_path_call_count()
            stats['notes'] = f'CBS exceeded max CT nodes ({MAX_CT_NODES})'
            return None, stats

        node = heapq.heappop(open_queue)
        ct_nodes += 1

        # CBS: 仅检测空间冲突（skip_time=True）
        conflict = detect_conflicts(node.solution, skip_time=True)

        print(f"[CBS] CT节点 #{ct_nodes}, cost={node.cost}, "
              f"约束数={len(node.constraints)}, 冲突={conflict}")

        if conflict is None:
            elapsed = time.perf_counter() - t_start
            metrics = _solution_metrics(node.solution)
            stats.update({
                'success': True,
                'total_time_s': round(elapsed, 4),
                'ct_nodes': ct_nodes,
                'find_path_calls': get_find_path_call_count(),
                'per_uav_path_length': metrics['per_uav_path_length'],
                'total_path_length': metrics['total_path_length'],
                'min_pairwise_dist': metrics['min_pairwise_dist'],
                'per_uav_arrival_time': metrics['per_uav_arrival_time'],
                'arrival_time_std': metrics['arrival_time_std'],
                'notes': 'CBS success',
            })
            print(f"[CBS] 找到无冲突解！耗时 {elapsed:.3f}s, CT节点 {ct_nodes}, "
                  f"低层调用 {stats['find_path_calls']}次")
            return node.solution, stats

        # --- 仅处理 vertex 和 edge 冲突 ---
        if conflict['type'] == 'vertex':
            agent1 = conflict['agent1']
            agent2 = conflict['agent2']
            time_t = conflict['time']
            pos1 = conflict['pos1']
            pos2 = conflict['pos2']

            # 分支1：约束 agent1
            new_c1 = copy.deepcopy(node.constraints)
            new_c1.append(Constraint(agent1, pos1, time_t))
            new_sol1 = copy.deepcopy(node.solution)
            new_path1 = find_path(start_list[agent1], goal_list[agent1],
                                  new_c1, agent1)
            if new_path1 is not None:
                new_sol1[agent1] = new_path1
                new_cost1 = sum(len(p) for p in new_sol1)
                heapq.heappush(open_queue,
                    CTNode(new_sol1, new_cost1, new_c1))

            # 分支2：约束 agent2
            new_c2 = copy.deepcopy(node.constraints)
            new_c2.append(Constraint(agent2, pos2, time_t))
            new_sol2 = copy.deepcopy(node.solution)
            new_path2 = find_path(start_list[agent2], goal_list[agent2],
                                  new_c2, agent2)
            if new_path2 is not None:
                new_sol2[agent2] = new_path2
                new_cost2 = sum(len(p) for p in new_sol2)
                heapq.heappush(open_queue,
                    CTNode(new_sol2, new_cost2, new_c2))

        elif conflict['type'] == 'edge':
            agent1 = conflict['agent1']
            agent2 = conflict['agent2']
            time_t = conflict['time']
            pos_i_t = conflict['pos_i_t']
            pos_i_t1 = conflict['pos_i_t1']
            pos_j_t = conflict['pos_j_t']
            pos_j_t1 = conflict['pos_j_t1']

            # 分支1：约束 agent1
            new_c1 = copy.deepcopy(node.constraints)
            new_c1.append(Constraint(agent1, pos_i_t, time_t))
            new_c1.append(Constraint(agent1, pos_i_t1, time_t + 1))
            new_sol1 = copy.deepcopy(node.solution)
            new_path1 = find_path(start_list[agent1], goal_list[agent1],
                                  new_c1, agent1)
            if new_path1 is not None:
                new_sol1[agent1] = new_path1
                new_cost1 = sum(len(p) for p in new_sol1)
                heapq.heappush(open_queue,
                    CTNode(new_sol1, new_cost1, new_c1))

            # 分支2：约束 agent2
            new_c2 = copy.deepcopy(node.constraints)
            new_c2.append(Constraint(agent2, pos_j_t, time_t))
            new_c2.append(Constraint(agent2, pos_j_t1, time_t + 1))
            new_sol2 = copy.deepcopy(node.solution)
            new_path2 = find_path(start_list[agent2], goal_list[agent2],
                                  new_c2, agent2)
            if new_path2 is not None:
                new_sol2[agent2] = new_path2
                new_cost2 = sum(len(p) for p in new_sol2)
                heapq.heappush(open_queue,
                    CTNode(new_sol2, new_cost2, new_c2))

    elapsed = time.perf_counter() - t_start
    stats['total_time_s'] = round(elapsed, 4)
    stats['ct_nodes'] = ct_nodes
    stats['find_path_calls'] = get_find_path_call_count()
    stats['notes'] = 'CBS open queue exhausted'
    return None, stats


def run_single_astar_baseline(start_list, goal_list):
    """对比基线：对每架 UAV 独立调用 sparse_astar.find_path（忽略所有协同约束），
    统计路径长度和冲突数。

    Returns:
        (solution, stats_dict)
    """
    from conflict import count_all_conflicts

    stats = {
        'success': False,
        'total_time_s': 0.0,
        'ct_nodes': 0,
        'find_path_calls': 0,
        'per_uav_path_length': [],
        'total_path_length': 0.0,
        'min_pairwise_dist': 0.0,
        'per_uav_arrival_time': [],
        'arrival_time_std': 0.0,
        'conflict_count': 0,
        'notes': '',
    }

    reset_find_path_call_count()
    t_start = time.perf_counter()

    solution = []
    for agent_id in range(len(start_list)):
        path = find_path(start_list[agent_id], goal_list[agent_id], [], agent_id)
        if path is None:
            elapsed = time.perf_counter() - t_start
            stats['total_time_s'] = round(elapsed, 4)
            stats['find_path_calls'] = get_find_path_call_count()
            stats['notes'] = f'baseline: no path for agent {agent_id}'
            return None, stats
        solution.append(path)

    elapsed = time.perf_counter() - t_start
    metrics = _solution_metrics(solution)
    conflicts = count_all_conflicts(solution)

    stats.update({
        'success': True,
        'total_time_s': round(elapsed, 4),
        'find_path_calls': get_find_path_call_count(),
        'per_uav_path_length': metrics['per_uav_path_length'],
        'total_path_length': metrics['total_path_length'],
        'min_pairwise_dist': metrics['min_pairwise_dist'],
        'per_uav_arrival_time': metrics['per_uav_arrival_time'],
        'arrival_time_std': metrics['arrival_time_std'],
        'conflict_count': conflicts['total'],
        'notes': f"baseline conflicts: {conflicts}",
    })
    return solution, stats
