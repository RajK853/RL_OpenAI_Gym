import numpy as np
from .base_policy import BasePolicy


class GreedyEpsilonPolicy(BasePolicy):

    def __init__(self, *, eps_range, eps_decay=0.01, explore_ratio=0.60, explore_exploit_interval=20, **kwargs):
        super(GreedyEpsilonPolicy, self).__init__(**kwargs)
        self.max_eps, self.min_eps = eps_range
        self.eps_decay = eps_decay
        self.explore_ratio = explore_ratio
        self.explore_exploit_interval = explore_exploit_interval
        self.last_epoch = -1
        self.eps_mask = 1
        self._eps = self.max_eps

    @property
    def eps(self):
        """
        Return epsilon value for the given epoch
        args:
            epoch (int): Epoch number
        returns:
            float : Epsilon value (_eps * eps_mask)
        """
        return self._eps * self.eps_mask

    def update_eps(self, epoch):
        """
        Update epsilon value and mask for the given epoch
        args:
            epoch (int) : Training epoch value
        """
        if epoch > self.last_epoch:  # Update values only in new epochs
            explore = (epoch % self.explore_exploit_interval) < (self.explore_ratio * self.explore_exploit_interval)
            self.eps_mask = 1 if explore else 0
            self._eps = max(self.min_eps, self._eps * np.math.e**(-self.eps_decay))
            self.last_epoch = epoch

    def _action(self, sess, states, **kwargs):
        """
        Get actions for given states from the given estimator
        args:
            states (Unknown) : States of the environment
        returns:
            list : Actions for the given states
        """
        actions = []
        estimator = kwargs["estimator"]
        for state in states:
            if np.random.random() < self.eps:
                action = self.action_space.sample()
            else:
                q_sa_values = estimator.predict(sess, np.array([state]))[0]
                action = np.argmax(q_sa_values)
            actions.append(action)
        return actions

    def hook_after_epoch(self, **kwargs):
        self.update_eps(kwargs["epoch"])

