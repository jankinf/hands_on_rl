'''
[
    ["PAD", "PAD", "PA1", "PA2", "PA3", "PA4", "RA1", "RA2", "RA3", "EOS", "EOS"],
    ["PAD", "PAD", "PAD", "PB1", "PB2", "PB3", "RB1", "RB2", "RB3", "RB4", "EOS"],
    ["PAD", "PAD", "PAD", "PAD", "PC1", "PC2", "RC1", "RC2", "RC3", "RC4", "RC5"],
]
只有EOS前一个token计算reward+kl，其他所有response tokens只计算kl
'''

def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
    """
    reward_function：计算最终的reward分数
    复习一下几个相关参数的默认值：
    self.kl_ctl = 0.1
    self.clip_reward_value = 5

    对于batch中的某个prompt来说，它最终的reward分数为：
    (1) 先计算actor和ref_model的logit相似度： -self.kl_ctl * (log_probs - ref_log_probs)
        其实写成self.kl_ctl * (ref_log_probs - log_probs)更好理解些
        这个值越大，说明ref_model对actor生成的结果的认可度越高（即表明rlhf没有训歪），
        没有训歪的情况下我们也应该给模型一些奖励，这个奖励就是self.kl_ctl * (ref_log_probs - log_probs)

    （2）由于我们只取最后一个token对应位置的分数作为reward_score，因此我们只需要：
        self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score

     (3) 同时我们对reward_score也做了大小限制，最大不超过self.clip_reward_value（超过统一给成self.clip_reward_value），
         最小不低于-self.clip_reward_value（低于统一给成-self.clip_reward_value）

     (4) 最后返回的rewards大小为：（batch_size, 各条数据的长度），对batch中的每条数据来说：
         - response的最后一位：self.kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score
         - response的其余位置：self.kl_ctl * (ref_log_probs - log_probs)

    """

    kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
    rewards = kl_divergence_estimate
    # ---------------------------------------------------------------------------------------------------
    # response开始的位置
    # （因为我们对prompt做过padding处理，因此batch中每个prompt长度一致，也就意味着每个response开始的位置一致）
    # （所以这里start是不加s的，只是一个int）
    # ---------------------------------------------------------------------------------------------------
    start = prompts.shape[1] - 1
    # ---------------------------------------------------------------------------------------------------
    # response结束的位置
    # （因为一个batch中，每个response的长度不一样，所以response的结束位置也不一样）
    # （所以这里end是加s的，ends的尺寸是(batch_size,)
    # ---------------------------------------------------------------------------------------------------
    ends = start + action_mask[:, start:].sum(1) + 1
    # ---------------------------------------------------------------------------------------------------
    # 对rewards_score做限制
    # ---------------------------------------------------------------------------------------------------
    reward_clip = torch.clamp(
        reward_score, -self.clip_reward_value, self.clip_reward_value
    )
    batch_size = log_probs.shape[0]
    for j in range(batch_size):
        rewards[j, start : ends[j]][-1] += reward_clip[j]  #

    return rewards
