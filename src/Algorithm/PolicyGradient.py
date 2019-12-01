import numpy as np
from .RLAlgorithm import RLAlgorithm
from src.Utils import normalize_array


class PolicyGradient(RLAlgorithm):

    def __init__(self, *, gamma=0.99, num_train=3, **kwargs):
        super(PolicyGradient, self).__init__(**kwargs)
        self.gamma = gamma
        self._policy_losses = []
        self.trajectory = []
        self.num_train = num_train

    @property
    def epoch_length(self):
        return len(self.trajectory)

    @property
    def sample_trajectory(self):
        states, actions, rewards = [], [], []
        for (state, action, reward) in self.trajectory:
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
        return np.array(states), np.array(actions), np.array(rewards)

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        state, action, reward, next_state, done = self.transition
        if self.training:
            self.trajectory.append((state, action, reward))
            if done:
                states, actions, rewards = self.sample_trajectory
                discount = self.gamma**np.arange(self.epoch_length, 0, -1)
                discounted_reward = discount*rewards
                discounted_return = np.cumsum(discounted_reward[::-1], axis=0)
                discounted_return = normalize_array(discounted_return)
                mean_policy_loss = np.mean([self.policy.update(self.sess, states, actions, discounted_return)
                                            for _ in range(self.num_train)])
                self._policy_losses.append(mean_policy_loss)
                self.trajectory.clear()