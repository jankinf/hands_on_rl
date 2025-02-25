'''
compute_reward只是计算出r_t（即时奖励），adv和returns都需要基于r_t计算
'''
def get_advantages_and_returns(self, values, rewards, start):
    """
    Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134

    没有引入GAE前的t时刻的优势值：
    detal_t = r_t + gamma * V_t+1 - V_t
    其中：
        - r_t表示t时刻的即时收益
        - V_t+1表示未来时刻的预期收益
        - r_t + gamma * V_t+1可理解成t时刻的实际预期收益
        - V_t可理解成t时刻的预估预期收益（是模型，例如critic model自己估算出来的）

    引入GAE后的t时刻的优势值：
    A_t = delta_t + gamma * lambda * A_t+1
    粗暴理解为在t时刻时，不仅考虑当下优势，还考虑了未来的优势
    为了知道A_t, 我们得知道A_t+1，所以在本算法中采取了从后往前做动态规划求解的方法，也即：
    假设T是最后一个时刻，则有A_T+1 = 0, 所以有: A_T = delta_T
    知道了A_T, 就可以依次往前倒推，把A_t-1, A_t-2之类都算出来了

    引入GAE后t时刻的实际预期收益
    returns_t = A_t + V_t
              = delta_t + gamma * lambda * A_t+1 + V_t
              = r_t + gamma * V_t+1 - V_t + gamma * lambda * A_t+1 + V_t
              = r_t + gamma * (V_t+1 + lambda * A_t+1)

    注意，这里不管是advantages还是returns，都只算response的部分
    """

    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    # 注意这里用了reversed，是采取从后往前倒推计算的方式
    for t in reversed(range(start, length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
        lastgaelam = delta + self.gamma * self.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)  # 优势
    returns = advantages + values[:, start:]  # 实际收益
    # values: 预期收益
    return advantages.detach(), returns
