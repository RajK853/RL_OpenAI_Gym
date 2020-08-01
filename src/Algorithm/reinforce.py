import numpy as np
from .rl_algorithm import RLAlgorithm
from src.utils import standardize_array


class Reinforce(RLAlgorithm):

    def __init__(self, *, gamma=0.99, num_train=1, **kwargs):
        super(Reinforce, self).__init__(**kwargs)
        self.gamma = gamma
        self.trajectory = []
        self.num_train = num_train
        self.mean_policy_loss = 0.0
        self.scalar_summaries += ("mean_policy_loss", )

    def sample_trajectory(self):
        states, actions, rewards = [], [], []
        for (state, action, reward) in self.trajectory:
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
        return np.array(states), np.array(actions), np.array(rewards)

    def compute_discounted_return(self, rewards):
        discounted_return = np.zeros(self.epoch_length, np.float32)
        sum_reward = 0
        for i in reversed(range(self.epoch_length)):
            sum_reward = rewards[i] + self.gamma * sum_reward
            discounted_return[i] = sum_reward
        return discounted_return

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training:
            state, action, reward, next_state, done = self.transition
            self.trajectory.append((state, action, reward))

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            states, actions, rewards = self.sample_trajectory()
            discounted_return = self.compute_discounted_return(rewards)
            discounted_return = standardize_array(discounted_return)
            self.mean_policy_loss = np.mean([self.policy.update(self.sess, states, actions, targets=discounted_return)
                                             for _ in range(self.num_train)])
            self.add_summaries()
            self.trajectory.clear()
