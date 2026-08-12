import math
import itertools

from config import D_SAFE

# 最大时间差（路径长度差），单位：步数（即时间步）
T_TAR = 1


def detect_conflicts(paths, D_SAFE=D_SAFE, T_TAR=T_TAR, skip_time=False):
    """Detect the first conflict among UAV paths.

    Args:
        paths: list of paths, each path is a list of (x, y, z, time) tuples.
        D_SAFE: minimum safe Euclidean distance.
        T_TAR: maximum allowed time difference (path length difference).
        skip_time: if True, skip time conflict detection (for CBS baseline).

    Returns:
        A conflict dictionary or None if no conflict is found.
    """
    if not paths:
        return None

    # --- 0. 路径交叉检测（快速启发式） ---
    # 计算每对路径的 bounding box，若不相交则跳过精细检测
    bboxes = []
    for path in paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        bboxes.append((min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))

    def bboxes_overlap(b1, b2, margin=D_SAFE):
        return (
            b1[0] - margin <= b2[1] and b2[0] - margin <= b1[1] and
            b1[2] - margin <= b2[3] and b2[2] - margin <= b1[3] and
            b1[4] - margin <= b2[5] and b2[4] - margin <= b1[5]
        )

    max_length = max(len(path) for path in paths)

    # 1. 点冲突（Vertex Conflict）：最早出现的时间步，任意两架 UAV 距离小于 D_SAFE
    for t in range(max_length):
        for i in range(len(paths)):
            if t >= len(paths[i]):
                continue
            pos_i = paths[i][t][:3]
            for j in range(i + 1, len(paths)):
                if t >= len(paths[j]):
                    continue
                # Bounding box 快速跳过
                if not bboxes_overlap(bboxes[i], bboxes[j]):
                    continue
                pos_j = paths[j][t][:3]
                if math.dist(pos_i, pos_j) < D_SAFE - 1e-9:
                    return {
                        'type': 'vertex',
                        'agent1': i,
                        'agent2': j,
                        'pos1': pos_i,
                        'pos2': pos_j,
                        'time': t,
                    }

    # 2. 时间冲突（Time Conflict）：路径长度差值超过 T_TAR
    if not skip_time:
        l_max = max_length
        for i, path in enumerate(paths):
            l_i = len(path)
            if l_max - l_i > T_TAR:
                return {
                    'type': 'time',
                    'agent': i,
                    'l_max': l_max,
                    'l_i': l_i,
                }

    # 3. 边冲突（Edge Conflict）：两架 UAV 在连续两个时间步之间交换位置（对穿）
    # 检查 (i, t) -> (j, t+1) 与 (j, t) -> (i, t+1) 是否同时发生
    for t in range(max_length - 1):
        for i in range(len(paths)):
            if t >= len(paths[i]) or t + 1 >= len(paths[i]):
                continue
            pi_t = paths[i][t][:3]
            pi_t1 = paths[i][t + 1][:3]
            for j in range(i + 1, len(paths)):
                if t >= len(paths[j]) or t + 1 >= len(paths[j]):
                    continue
                pj_t = paths[j][t][:3]
                pj_t1 = paths[j][t + 1][:3]
                # 检查是否位置互换（或边交叉导致间距 < D_SAFE）
                # 条件1: i 从 t 到 t+1 的位移与 j 从 t 到 t+1 的位移形成交叉
                if math.dist(pi_t, pj_t1) < D_SAFE - 1e-9 and math.dist(pi_t1, pj_t) < D_SAFE - 1e-9:
                    return {
                        'type': 'edge',
                        'agent1': i,
                        'agent2': j,
                        'pos_i_t': pi_t,
                        'pos_i_t1': pi_t1,
                        'pos_j_t': pj_t,
                        'pos_j_t1': pj_t1,
                        'time': t,
                    }

    return None


def count_all_conflicts(paths, D_SAFE=D_SAFE, T_TAR=T_TAR):
    """Count ALL conflicts in a path set (for baseline evaluation).

    Returns:
        dict with 'vertex', 'edge', 'time' counts and total.
    """
    counts = {'vertex': 0, 'edge': 0, 'time': 0, 'total': 0}
    if not paths:
        return counts

    max_length = max(len(path) for path in paths)

    # Vertex conflicts
    for t in range(max_length):
        for i in range(len(paths)):
            if t >= len(paths[i]):
                continue
            pos_i = paths[i][t][:3]
            for j in range(i + 1, len(paths)):
                if t >= len(paths[j]):
                    continue
                pos_j = paths[j][t][:3]
                if math.dist(pos_i, pos_j) < D_SAFE - 1e-9:
                    counts['vertex'] += 1

    # Edge conflicts
    for t in range(max_length - 1):
        for i in range(len(paths)):
            if t >= len(paths[i]) or t + 1 >= len(paths[i]):
                continue
            pi_t = paths[i][t][:3]
            pi_t1 = paths[i][t + 1][:3]
            for j in range(i + 1, len(paths)):
                if t >= len(paths[j]) or t + 1 >= len(paths[j]):
                    continue
                pj_t = paths[j][t][:3]
                pj_t1 = paths[j][t + 1][:3]
                if math.dist(pi_t, pj_t1) < D_SAFE - 1e-9 and math.dist(pi_t1, pj_t) < D_SAFE - 1e-9:
                    counts['edge'] += 1

    # Time conflicts
    l_max = max_length
    for i, path in enumerate(paths):
        if l_max - len(path) > T_TAR:
            counts['time'] += 1

    counts['total'] = counts['vertex'] + counts['edge'] + counts['time']
    return counts
