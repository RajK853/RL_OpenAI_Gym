import numpy as np
import tensorflow.compat.v1 as tf_v1
from gym.spaces import Discrete

from src.utils import get_space_size


class BasePolicy:

    def __init__(self, *, env, name="policy"):
        self.scope = name
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_size = get_space_size(self.action_space)
        self.observation_size = get_space_size(self.observation_space)
        self.discrete_action_space = isinstance(self.action_space, Discrete)
        self.scalar_summaries = ()
        self.histogram_summaries = ()
        self.summary_op = None
        self.summary = None
        self._loss = None
        self.train_op = None

    @property
    def loss(self):
        assert self._loss is not None, "Loss function not initialized!"
        return self._loss

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def obs_shape(self):
        return self.observation_space.shape

    @staticmethod
    def process_action(actions):
        actions = np.squeeze(actions)
        if not isinstance(actions, np.ndarray):
            return actions.item()
        return actions

    def reshape_state(self, states):
        states = states.reshape(-1, *self.observation_space.shape)
        return states

    def action(self, sess, states, **kwargs):
        states = self.reshape_state(states)
        actions = self._action(sess, states, **kwargs)
        actions = self.process_action(actions)
        return actions

    def _action(self, sess, states, **kwargs):
        raise NotImplementedError

    def update(self, sess, states, actions, **kwargs):
        return 0.0

    def hook_before_epoch(self, **kwargs):
        pass

    def hook_after_epoch(self, **kwargs):
        pass

    def init_summaries(self, tag="", force=False):
        if self.summary_op is None or force:
            _summaries = []
            for summary_type in ("scalar", "histogram"):
                summary_func = getattr(tf_v1.summary, summary_type)
                for summary_attr in getattr(self, f"{summary_type}_summaries"):
                    attr = getattr(self, summary_attr)
                    _summaries.append(summary_func(f"{tag}/{self.scope}/{summary_attr}", attr))
            if _summaries:
                self.summary_op = tf_v1.summary.merge(_summaries)
