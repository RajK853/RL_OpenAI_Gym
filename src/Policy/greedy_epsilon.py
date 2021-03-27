import numpy as np
from .base_policy import BasePolicy
from src.utils import get_scheduler


class GreedyEpsilonPolicy(BasePolicy):

    def __init__(self, *, eps_kwargs, explore_ratio=0.60, explore_exploit_interval=20, **kwargs):
        super(GreedyEpsilonPolicy, self).__init__(**kwargs)
        self.eps_scheduler = get_scheduler(eps_kwargs)
        self.explore_ratio = explore_ratio
        self.explore_exploit_interval = explore_exploit_interval
        self.explore_interval = self.explore_ratio*self.explore_exploit_interval
        self.eps_mask = 1
        self.scalar_summaries += ("eps", )

    @property
    def eps(self):
        """
        Returns masked epsilon value for the current epoch; Epsilon value = eps_mask * eps
        """
        return self.eps_mask*self.eps_scheduler.value

    def _action(self, sess, states, deterministic=False, **kwargs):
        """
        Get actions for given states from the given estimator
        args:
            state (Unknown) : State of the environment
        returns:
            list : Action for the given state
        """
        estimator = kwargs["estimator"]
        if deterministic:
            q_values = estimator.predict(states)
            actions = np.argmax(q_values, axis=-1)
        else:
            actions = []
            for state in states:
                if np.random.random() < self.eps:
                    action = self.action_space.sample()
                else:
                    q_values = estimator.predict(state)[0]
                    action = np.argmax(q_values)
                actions.append(action)
        return actions

    def hook_after_epoch(self, **kwargs):
        self.eps_scheduler.increment()
        self.eps_mask = (self.eps_scheduler.steps % self.explore_exploit_interval) < self.explore_interval
