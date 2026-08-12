"""
visualize.py — 多无人机路径规划可视化、批量实验与 CSV 输出。

用法：
    python visualize.py          # 交互式可视化（默认 parallel_3 场景）
    python visualize.py --batch  # 批量实验模式，输出 results.csv
"""

import csv
import math
import os
import random
import sys
import time

try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

from mcbs import mcbs, cbs, run_single_astar_baseline
from config import D_SAFE, CSV_COLUMNS

# CSV 输出路径
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')


def plot_paths(solution, title="MCBS Paths"):
    """Plot UAV paths in XY plane."""
    if not HAS_PLT:
        print("(matplotlib not installed, skipping plot)")
        return
    if solution is None:
        print("No solution to plot.")
        return

    colors = ["blue", "orange", "green", "purple", "cyan", "magenta"]
    plt.figure(figsize=(8, 8))

    for idx, path in enumerate(solution):
        xs = [point[0] for point in path]
        ys = [point[1] for point in path]
        color = colors[idx % len(colors)]
        plt.plot(xs, ys, marker="o", markersize=2, color=color, label=f"UAV {idx}")
        plt.plot(xs[0], ys[0], marker="*", color="green", markersize=12,
                 label=f"Start {idx}" if idx == 0 else "")
        plt.plot(xs[-1], ys[-1], marker="*", color="red", markersize=12,
                 label=f"Goal {idx}" if idx == 0 else "")

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.show()


def _make_row(algorithm, num_agents, starts, goals, d_safe, t_tar,
              success, stats, extra_notes=''):
    """将实验数据打包为一行 CSV 字典。"""
    return {
        'algorithm': algorithm,
        'num_agents': num_agents,
        'starts': str(starts),
        'goals': str(goals),
        'D_SAFE': d_safe,
        'T_TAR': t_tar,
        'success': success,
        'total_time_s': stats.get('total_time_s', ''),
        'ct_nodes': stats.get('ct_nodes', ''),
        'find_path_calls': stats.get('find_path_calls', ''),
        'per_uav_path_length': str(stats.get('per_uav_path_length', [])),
        'total_path_length': stats.get('total_path_length', ''),
        'min_pairwise_dist': stats.get('min_pairwise_dist', ''),
        'per_uav_arrival_time': str(stats.get('per_uav_arrival_time', [])),
        'arrival_time_std': stats.get('arrival_time_std', ''),
        'conflict_count': stats.get('conflict_count', ''),
        'notes': (stats.get('notes', '') + '; ' + extra_notes).strip('; '),
    }


def write_csv_header(filepath):
    """如果 CSV 文件不存在，写入表头。"""
    if not os.path.exists(filepath):
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def append_csv_row(filepath, row_dict):
    """追加一行到 CSV 文件。"""
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row_dict)


def generate_random_scenarios(num_scenarios, num_agents, map_size=(100, 100, 50),
                              min_separation=20, seed=42):
    """生成随机起点-终点场景。

    Args:
        num_scenarios: 场景数量
        num_agents: 每场景的无人机数量
        map_size: (x_max, y_max, z_max)
        min_separation: 起点与终点之间最小距离
        seed: 随机种子

    Returns:
        list of (start_list, goal_list)
    """
    rng = random.Random(seed)
    scenarios = []
    x_max, y_max, z_max = map_size

    for _ in range(num_scenarios):
        starts = []
        goals = []
        for _ in range(num_agents):
            while True:
                sx, sy = rng.randint(10, x_max - 10), rng.randint(10, y_max - 10)
                sz = rng.randint(0, min(20, z_max))
                gx, gy = rng.randint(10, x_max - 10), rng.randint(10, y_max - 10)
                gz = rng.randint(0, min(20, z_max))
                # 确保起点和终点之间有足够距离
                if math.dist((sx, sy, sz), (gx, gy, gz)) >= min_separation:
                    starts.append((sx, sy, sz))
                    goals.append((gx, gy, gz))
                    break
        scenarios.append((starts, goals))
    return scenarios


def run_batch(scenarios, csv_path=CSV_PATH, d_safe=D_SAFE, t_tar=1):
    """批量运行 MCBS 和 Sparse_Astar_Only 两种算法，结果追加到 CSV。

    Args:
        scenarios: list of (start_list, goal_list)
        csv_path: CSV 输出路径
        d_safe: 安全距离参数
        t_tar: 时间差阈值
    """
    from conflict import T_TAR as default_t_tar
    if t_tar is None:
        t_tar = default_t_tar

    write_csv_header(csv_path)
    total = len(scenarios)

    for idx, (starts, goals) in enumerate(scenarios):
        num_agents = len(starts)
        print(f"\n{'='*60}")
        print(f"场景 {idx + 1}/{total}: {num_agents} 架 UAV")
        print(f"  起点: {starts}")
        print(f"  终点: {goals}")
        print(f"{'='*60}")

        # ---- MCBS ----
        print("\n--- MCBS ---")
        t0 = time.perf_counter()
        solution, stats = mcbs(starts, goals)
        t1 = time.perf_counter()
        success = stats.get('success', False)
        print(f"MCBS 结果: {'成功' if success else '失败'} ({t1 - t0:.2f}s)")
        row = _make_row('MCBS', num_agents, starts, goals, d_safe, t_tar,
                        success, stats)
        append_csv_row(csv_path, row)

        # ---- Sparse A* Only (baseline) ----
        print("\n--- Sparse_Astar_Only ---")
        t0 = time.perf_counter()
        sol_bl, stats_bl = run_single_astar_baseline(starts, goals)
        t1 = time.perf_counter()
        success_bl = stats_bl.get('success', False)
        if sol_bl is not None:
            print(f"Baseline 结果: {num_agents} 条路径, "
                  f"冲突数={stats_bl.get('conflict_count', '?')} ({t1 - t0:.2f}s)")
        else:
            print(f"Baseline 失败: {stats_bl.get('notes', '?')} ({t1 - t0:.2f}s)")
        row_bl = _make_row('Sparse_Astar_Only', num_agents, starts, goals,
                           d_safe, t_tar, success_bl, stats_bl)
        append_csv_row(csv_path, row_bl)

        # ---- CBS (仅空间冲突) ----
        print("\n--- CBS (空间冲突) ---")
        t0 = time.perf_counter()
        sol_cbs, stats_cbs = cbs(starts, goals)
        t1 = time.perf_counter()
        success_cbs = stats_cbs.get('success', False)
        print(f"CBS 结果: {'成功' if success_cbs else '失败'} ({t1 - t0:.2f}s)")
        row_cbs = _make_row('CBS', num_agents, starts, goals, d_safe, t_tar,
                            success_cbs, stats_cbs)
        append_csv_row(csv_path, row_cbs)

    print(f"\n全部 {total} 个场景完成，结果已保存至 {csv_path}")


# ===================== 预设场景 =====================

PRESET_SCENARIOS = {
    "crossing_2": {
        "starts": [(20, 20, 0), (80, 80, 0)],
        "goals": [(80, 80, 0), (20, 20, 0)],
    },
    "crossing_3": {
        "starts": [(20, 20, 0), (80, 20, 0), (50, 80, 0)],
        "goals": [(80, 80, 0), (20, 80, 0), (50, 20, 0)],
    },
    "parallel_2": {
        "starts": [(20, 30, 0), (20, 70, 0)],
        "goals": [(80, 30, 0), (80, 70, 0)],
    },
    "parallel_3": {
        "starts": [(20, 20, 0), (20, 50, 0), (20, 80, 0)],
        "goals": [(80, 20, 0), (80, 50, 0), (80, 80, 0)],
    },
    "random_2_seed0": {
        "starts": [(15, 30, 0), (70, 60, 0)],
        "goals": [(85, 70, 0), (25, 25, 0)],
    },
}


def get_preset_scenarios():
    """返回预设场景列表。"""
    sc_list = []
    for name, sc in PRESET_SCENARIOS.items():
        sc_list.append((name, sc['starts'], sc['goals']))
    return sc_list


def run_presets(csv_path=CSV_PATH):
    """运行所有预设场景。"""
    from conflict import T_TAR
    write_csv_header(csv_path)

    for name, starts, goals in get_preset_scenarios():
        num_agents = len(starts)
        print(f"\n{'='*60}")
        print(f"预设场景: {name} ({num_agents} UAVs)")
        print(f"{'='*60}")

        # MCBS
        print("\n--- MCBS ---")
        solution, stats = mcbs(starts, goals)
        success = stats.get('success', False)
        print(f"MCBS: {'成功' if success else '失败'}")
        row = _make_row('MCBS', num_agents, starts, goals, D_SAFE, T_TAR,
                        success, stats)
        append_csv_row(csv_path, row)

        # Baseline
        print("\n--- Sparse_Astar_Only ---")
        sol_bl, stats_bl = run_single_astar_baseline(starts, goals)
        success_bl = stats_bl.get('success', False)
        print(f"Baseline: {'成功' if success_bl else '失败'}, "
              f"冲突数={stats_bl.get('conflict_count', '?')}")
        row_bl = _make_row('Sparse_Astar_Only', num_agents, starts, goals,
                           D_SAFE, T_TAR, success_bl, stats_bl)
        append_csv_row(csv_path, row_bl)

        # CBS (仅空间冲突)
        print("\n--- CBS (空间冲突) ---")
        sol_cbs, stats_cbs = cbs(starts, goals)
        success_cbs = stats_cbs.get('success', False)
        print(f"CBS: {'成功' if success_cbs else '失败'}, "
              f"耗时={stats_cbs.get('total_time_s', '?')}s, "
              f"CT节点={stats_cbs.get('ct_nodes', '?')}")
        row_cbs = _make_row('CBS', num_agents, starts, goals, D_SAFE, T_TAR,
                            success_cbs, stats_cbs)
        append_csv_row(csv_path, row_cbs)

    print(f"\n预设场景完成，结果保存至 {csv_path}")


# ===================== main =====================

if __name__ == "__main__":
    print("脚本开始运行...", flush=True)

    if '--batch' in sys.argv:
        # 批量模式：预设 + 随机场景
        print("批量实验模式")
        run_presets(CSV_PATH)

        # 生成随机场景并测试
        random_scenarios = generate_random_scenarios(
            num_scenarios=5, num_agents=2, seed=42)
        run_batch(random_scenarios, CSV_PATH)

        random_scenarios_3 = generate_random_scenarios(
            num_scenarios=3, num_agents=3, seed=123)
        run_batch(random_scenarios_3, CSV_PATH)

    elif '--presets' in sys.argv:
        # 仅预设场景
        run_presets(CSV_PATH)

    elif '--random' in sys.argv:
        # 仅随机场景
        n = 5
        n_agents = 2
        for arg in sys.argv:
            if arg.startswith('--n='):
                n = int(arg.split('=')[1])
            if arg.startswith('--agents='):
                n_agents = int(arg.split('=')[1])
        random_scenarios = generate_random_scenarios(
            num_scenarios=n, num_agents=n_agents)
        run_batch(random_scenarios, CSV_PATH)

    else:
        # ===== 交互模式：默认使用 parallel_3 场景，保证快速成功 =====
        scene = PRESET_SCENARIOS["parallel_3"]
        start_list = scene["starts"]
        goal_list = scene["goals"]
        num_agents = len(start_list)

        print(f"\n默认场景: parallel_3 ({num_agents} 架无人机平行飞行)")
        print(f"  起点: {start_list}")
        print(f"  终点: {goal_list}")

        # ---- MCBS ----
        print("\n" + "=" * 60)
        print("=== MCBS (协同规划) ===")
        print("=" * 60)
        solution, stats = mcbs(start_list, goal_list)
        success = stats.get('success', False)

        if success:
            print(f"\n[MCBS 成功] 总路径长度={stats['total_path_length']:.2f}, "
                  f"耗时={stats['total_time_s']:.3f}s, "
                  f"CT节点={stats['ct_nodes']}, "
                  f"低层调用={stats['find_path_calls']}")
            print(f"  各UAV路径长度: {[f'{x:.1f}' for x in stats['per_uav_path_length']]}")
            print(f"  最小两两距离: {stats['min_pairwise_dist']:.3f}")
            print(f"  到达时间: {stats['per_uav_arrival_time']}")
            print(f"  到达时间标准差: {stats['arrival_time_std']:.3f}")
            plot_paths(solution, title=f"MCBS — parallel_3 ({num_agents} UAVs, D_SAFE={D_SAFE})")
        else:
            print(f"\n[MCBS 失败] 耗时={stats['total_time_s']:.3f}s, "
                  f"CT节点={stats['ct_nodes']}, "
                  f"低层调用={stats['find_path_calls']}, "
                  f"原因: {stats.get('notes', 'unknown')}")

        # ---- Sparse A* Only (baseline) ----
        print("\n" + "=" * 60)
        print("=== Sparse A* Only (无协同基线) ===")
        print("=" * 60)
        sol_bl, stats_bl = run_single_astar_baseline(start_list, goal_list)
        success_bl = stats_bl.get('success', False)

        if success_bl:
            print(f"\n[Baseline 成功] 总路径长度={stats_bl['total_path_length']:.2f}, "
                  f"冲突数={stats_bl['conflict_count']}, "
                  f"耗时={stats_bl['total_time_s']:.3f}s")
            print(f"  各UAV路径长度: {[f'{x:.1f}' for x in stats_bl['per_uav_path_length']]}")
            print(f"  最小两两距离: {stats_bl['min_pairwise_dist']:.3f}")
            print(f"  到达时间标准差: {stats_bl['arrival_time_std']:.3f}")
            plot_paths(sol_bl, title=f"Sparse A* Only — parallel_3 (No Coordination, {stats_bl['conflict_count']} conflicts)")
        else:
            print(f"\n[Baseline 失败] 耗时={stats_bl['total_time_s']:.3f}s, "
                  f"原因: {stats_bl.get('notes', 'unknown')}")

        # ---- CBS (仅空间冲突) ----
        print("\n" + "=" * 60)
        print("=== CBS (仅空间冲突) ===")
        print("=" * 60)
        sol_cbs, stats_cbs = cbs(start_list, goal_list)
        success_cbs = stats_cbs.get('success', False)

        if success_cbs:
            print(f"\n[CBS 成功] 总路径长度={stats_cbs['total_path_length']:.2f}, "
                  f"耗时={stats_cbs['total_time_s']:.3f}s, "
                  f"CT节点={stats_cbs['ct_nodes']}, "
                  f"低层调用={stats_cbs['find_path_calls']}")
            print(f"  各UAV路径长度: {[f'{x:.1f}' for x in stats_cbs['per_uav_path_length']]}")
            print(f"  最小两两距离: {stats_cbs['min_pairwise_dist']:.3f}")
            print(f"  到达时间: {stats_cbs['per_uav_arrival_time']}")
            print(f"  到达时间标准差: {stats_cbs['arrival_time_std']:.3f}")
            plot_paths(sol_cbs, title=f"CBS — parallel_3 ({num_agents} UAVs, D_SAFE={D_SAFE})")
        else:
            print(f"\n[CBS 失败] 耗时={stats_cbs['total_time_s']:.3f}s, "
                  f"CT节点={stats_cbs['ct_nodes']}, "
                  f"低层调用={stats_cbs['find_path_calls']}, "
                  f"原因: {stats_cbs.get('notes', 'unknown')}")
