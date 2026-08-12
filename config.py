"""Configuration constants for multi-UAV path planning."""

# 地图大小 (X, Y, Z)
MAP_SIZE = (100, 100, 50)

# 每步移动距离
STEP_SIZE = 2

# 最小转弯半径，必须大于等于 STEP_SIZE
R_MIN = 1.5   # 减小转弯半径 = 增大每步最大偏转角，提升机动性

# 最大爬升角，单位：度
THETA_MAX = 30

# 安全距离，两架无人机最小间距
D_SAFE = 3

# 速度，假设所有无人机匀速，时间 = 路径点索引
V = 1

# 高层搜索超时（秒）
TIMEOUT = 60  # 60 秒超时

# 约束树最大展开节点数
MAX_CT_NODES = 10000

# 低层 A* 最大搜索步数上限
LOW_LEVEL_MAX_STEPS = 50000

# 终点判定阈值：当 agent 距目标在此距离内即认为到达
# 与 STEP_SIZE 对齐，保证可达且不会过早截断
GOAL_THRESHOLD = 1.5 * STEP_SIZE

# CSV 输出列（供 visualize 使用）
CSV_COLUMNS = [
    "algorithm",
    "num_agents",
    "starts",
    "goals",
    "D_SAFE",
    "T_TAR",
    "success",
    "total_time_s",
    "ct_nodes",
    "find_path_calls",
    "per_uav_path_length",
    "total_path_length",
    "min_pairwise_dist",
    "per_uav_arrival_time",
    "arrival_time_std",
    "conflict_count",
    "notes",
]


class Node:
    """Path planning node for a UAV search graph."""

    def __init__(self, x, y, z, parent=None, g=0.0, h=0.0, yaw=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.parent = parent
        self.g = float(g)
        self.h = float(h)
        self.f = self.g + self.h
        self.yaw = float(yaw)

    def __repr__(self):
        return (
            f"Node(x={self.x}, y={self.y}, z={self.z}, g={self.g}, h={self.h}, "
            f"f={self.f}, yaw={self.yaw})"
        )

    def __lt__(self, other):
        """用于 heapq 优先队列的比较函数，当 f 值相同时比较 g 值（可选）。"""
        if self.f == other.f:
            return self.g < other.g
        return self.f < other.f


class Constraint:
    """Represents a time-space constraint for a single UAV."""

    def __init__(self, agent_id, pos, time):
        self.agent_id = agent_id
        self.pos = tuple(pos)
        self.time = time

    def __repr__(self):
        return (
            f"Constraint(agent_id={self.agent_id}, pos={self.pos}, time={self.time})"
        )


# 高层启发式惩罚因子
OMEGA_S = 10   # 空间冲突惩罚权重
OMEGA_T = 5    # 时间冲突惩罚权重
