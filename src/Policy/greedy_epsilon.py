import numpy as np
from .base_policy import BasePolicy
from src.utils import get_scheduler


class GreedyEpsilonPolicy(BasePolicy):

    def __init__(self, *, eps_kwargs, explore_ratio=0.60, explore_exploit_interval=20, **kwargs):
        super(GreedyEpsilonPolicy, self).__init__(**kwargs)
        self.eps_scheduler = get_scheduler(eps_kwargs)
        self.schedulers += (self.eps_scheduler, )
        self.scalar_summaries += ("eps", )
        ######################### Experimental feature ##########################
        """
        Idea:
        Instead of choosing actions greedily with some random actions based on the epsilon value, 
        we introduce some epoch intervals periodically where the policy operates deterministically
        by temporarily setting the epsilon value to 0. 
        """
        # TODO: Implement it as a scheduler in all policies
        self.explore_ratio = explore_ratio
        self.explore_exploit_interval = explore_exploit_interval
        self.explore_interval = self.explore_ratio*self.explore_exploit_interval
        self.eps_mask = 1
        #########################################################################

    @property
    def eps(self):
        """
        Returns masked epsilon value for the current epoch; Epsilon value = eps_mask * eps
        """
        return self.eps_mask*self.eps_scheduler.value

    def set_model(self, model):
        self.model = model

    def _action(self, sess, states, deterministic=False, **kwargs):
        if states.shape == self.obs_shape:
            states = np.expand_dims(states, axis=0)
        if deterministic or np.random.random() > self.eps:
            q_values = self.model.predict(states)
            actions = np.argmax(q_values, axis=-1)
        else:
            actions = [self.action_space.sample() for _ in states]
        return actions

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        self.eps_mask = (self.eps_scheduler.steps % self.explore_exploit_interval) < self.explore_interval
