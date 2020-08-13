import numpy as np
from . import OnPolicyAlgorithm
from src.utils import standardize_array


class Reinforce(OnPolicyAlgorithm):
    VALID_POLICIES = {"DiscretePolicy", "GaussianPolicy"}

    def __init__(self, *, gamma=0.99, num_train=1, **kwargs):
        super(Reinforce, self).__init__(**kwargs)
        self.gamma = gamma
        self.trajectory = {}
        self.num_train = num_train
        self.mean_policy_loss = 0.0
        self.field_names = ("state", "action", "reward")
        self.scalar_summaries += ("mean_policy_loss", )

    def compute_discounted_return(self, rewards):
        discounted_return = np.zeros(self.epoch_length, np.float32)
        sum_reward = 0
        for i in reversed(range(self.epoch_length)):
            sum_reward = rewards[i] + self.gamma * sum_reward
            discounted_return[i] = sum_reward
        return discounted_return

    def train(self):
        states, actions, rewards = self.sample_trajectory()
        discounted_return = self.compute_discounted_return(rewards)
        discounted_return = standardize_array(discounted_return)
        args = (self.sess, states, actions)
        self.mean_policy_loss = np.mean([self.policy.update(*args, targets=discounted_return)
                                         for _ in range(self.num_train)])

    def hook_before_train(self, **kwargs):
        self.policy.init_default_loss()
        super().hook_before_train(**kwargs)

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training:
            state, action, reward, *_ = self.transition
            self.add_transition(state=state, action=action, reward=reward)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.train()
            self.add_summaries()
            self.trajectory.clear()
