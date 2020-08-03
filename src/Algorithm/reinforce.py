import numpy as np
from .rl_algorithm import RLAlgorithm
from src.utils import standardize_array


class Reinforce(RLAlgorithm):

    def __init__(self, *, gamma=0.99, num_train=1, **kwargs):
        super(Reinforce, self).__init__(**kwargs)
        self.gamma = gamma
        self.trajectory = {}
        self.num_train = num_train
        self.mean_policy_loss = 0.0
        self.scalar_summaries += ("mean_policy_loss", )

    def add_transition(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.trajectory.keys():
                self.trajectory[key] = []
            self.trajectory[key].append(value)

    def sample_trajectory(self):
        return {key: np.array(value) for key, value in self.trajectory.items()}

    def compute_discounted_return(self, rewards):
        discounted_return = np.zeros(self.epoch_length, np.float32)
        sum_reward = 0
        for i in reversed(range(self.epoch_length)):
            sum_reward = rewards[i] + self.gamma * sum_reward
            discounted_return[i] = sum_reward
        return discounted_return

    def train(self):
        trajectory = self.sample_trajectory()
        discounted_return = self.compute_discounted_return(trajectory["reward"])
        discounted_return = standardize_array(discounted_return)
        args = (self.sess, trajectory["state"], trajectory["action"])
        self.mean_policy_loss = np.mean([self.policy.update(*args, targets=discounted_return)
                                         for _ in range(self.num_train)])

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training:
            state, action, reward, next_state, done = self.transition
            self.add_transition(state=state, action=action, reward=reward)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.train()
            self.add_summaries()
            self.trajectory.clear()
