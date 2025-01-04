import numpy as np

from .solver import EpsilonGreedy, DecayingEpsilonGreedy, UCB, ThompsonSampling
from .env.bernoulli_bandit import BernoulliBandit
from .utils.plot import plot_results


K = 10
np.random.seed(42)
bandit_10_arm = BernoulliBandit(K)

print("随机生成了一个%d臂伯努利老虎机" % K)
print(
    "获奖概率最大的拉杆为%d号,其获奖概率为%.4f"
    % (
        bandit_10_arm.best_idx,
        bandit_10_arm.best_prob,
    )
)

epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print("epsilon-贪婪算法的累积懊悔为：", epsilon_greedy_solver.regret)
plot_results(
    [epsilon_greedy_solver],
    ["EpsilonGreedy"],
    "basic/multi_armed_bandit/plots/multi_armed_bandit.png",
)


epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons]
epsilon_greedy_solver_names = ["epsilon={}".format(e) for e in epsilons]
for solver in epsilon_greedy_solver_list:
    solver.run(5000)

plot_results(
    epsilon_greedy_solver_list,
    epsilon_greedy_solver_names,
    "basic/multi_armed_bandit/plots/multi_armed_bandit_eps_ablation.png",
)


decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(5000)
print("epsilon值衰减的贪婪算法的累积懊悔为：", decaying_epsilon_greedy_solver.regret)
plot_results(
    [decaying_epsilon_greedy_solver],
    ["DecayingEpsilonGreedy"],
    "basic/multi_armed_bandit/plots/multi_armed_bandit_decaying_eps.png",
)


coef = 1  # 控制不确定性比重的系数
UCB_solver = UCB(bandit_10_arm, coef)
UCB_solver.run(5000)
print("上置信界算法的累积懊悔为：", UCB_solver.regret)
plot_results(
    [UCB_solver],
    ["UCB"],
    "basic/multi_armed_bandit/plots/multi_armed_bandit_ucb.png",
)


thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print("汤普森采样算法的累积懊悔为：", thompson_sampling_solver.regret)
plot_results(
    [thompson_sampling_solver],
    ["ThompsonSampling"],
    "basic/multi_armed_bandit/plots/multi_armed_bandit_ts.png",
)
