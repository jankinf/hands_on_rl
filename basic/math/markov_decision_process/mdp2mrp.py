import numpy as np

S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持s1", "前往s1", "前往s2", "前往s3", "前往s4", "前往s5", "概率前往"]  # 动作集合
# 状态转移函数 P(s'|s,a)
P = {
    "s1-保持s1-s1": 1.0,
    "s1-前往s2-s2": 1.0,
    "s2-前往s1-s1": 1.0,
    "s2-前往s3-s3": 1.0,
    "s3-前往s4-s4": 1.0,
    "s3-前往s5-s5": 1.0,
    "s4-前往s5-s5": 1.0,
    "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4,
    "s4-概率前往-s4": 0.4,
}
# 奖励函数 R(s, a)
R = {
    "s1-保持s1": -1,
    "s1-前往s2": 0,
    "s2-前往s1": -1,
    "s2-前往s3": -2,
    "s3-前往s4": -2,
    "s3-前往s5": 0,
    "s4-前往s5": 10,
    "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)

# 策略1,随机策略
Pi_1 = {
    "s1-保持s1": 0.5,
    "s1-前往s2": 0.5,
    "s2-前往s1": 0.5,
    "s2-前往s3": 0.5,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.5,
    "s4-概率前往": 0.5,
}
# 策略2
Pi_2 = {
    "s1-保持s1": 0.6,
    "s1-前往s2": 0.4,
    "s2-前往s1": 0.3,
    "s2-前往s3": 0.7,
    "s3-前往s4": 0.5,
    "s3-前往s5": 0.5,
    "s4-前往s5": 0.1,
    "s4-概率前往": 0.9,
}


# 把输入的两个字符串通过“-”连接,便于使用上述定义的P、R变量
def join(str1, str2):
    return str1 + "-" + str2


def compute_v(P, rewards, gamma, states_num):
    """
    给定MRP，计算价值函数，利用贝尔曼方程的矩阵形式计算解析解, states_num是MRP的状态数
    """
    rewards = np.array(rewards).reshape((-1, 1))  # 将rewards写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


def compute_state_transition_matrix(S, A, P, Pi):
    """
    计算从MDP转换到MRP的状态转移矩阵

    参数:
    S: 状态集合
    A: 动作集合
    P: MDP的状态转移函数字典 P(s'|s,a)
    Pi: 策略字典 Pi(a|s)

    返回:
    P_mrp: numpy数组，表示MRP的状态转移矩阵
    """
    n = len(S)  # 状态数量
    P_mrp = np.zeros((n, n))  # 初始化状态转移矩阵

    # 建立状态索引字典，方便后续使用
    state_to_idx = {state: idx for idx, state in enumerate(S)}

    # 遍历每个状态和动作
    for s in S:
        for a in A:
            # 获取当前状态-动作对的策略概率
            pi_key = join(s, a)
            if pi_key in Pi:
                pi_prob = Pi[pi_key]

                # 遍历所有可能的下一个状态
                for s_next in S:
                    # 构造状态转移的key
                    p_key = join(join(s, a), s_next)

                    # 如果存在这个转移概率
                    if p_key in P:
                        i = state_to_idx[s]  # 当前状态索引
                        j = state_to_idx[s_next]  # 下一个状态索引
                        # 更新转移矩阵
                        P_mrp[i][j] += pi_prob * P[p_key]

    return P_mrp


P_mrp = compute_state_transition_matrix(S, A, P, Pi_1)
print(P_mrp)


# MDP的状态价值函数可以转化为MRP的价值函数
gamma = 0.5
# 转化后的MRP的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]

V = compute_v(P_from_mdp_to_mrp, R_from_mdp_to_mrp, gamma, 5)
print("MDP中每个状态价值分别为\n", V)
