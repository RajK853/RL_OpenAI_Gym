import tensorflow.compat.v1 as tf_v1
from gym.spaces import Discrete

from src.utils import get_space_size


class BasePolicy:
    PARAMETERS = {"clip_norm"}

    def __init__(self, *, env, name="policy", clip_norm=2.0):
        self.scope = name
        self.clip_norm = clip_norm
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_size = get_space_size(self.action_space)
        self.observation_size = get_space_size(self.observation_space)
        self.discrete_action_space = isinstance(self.action_space, Discrete)
        self.schedulers = ()
        self.scalar_summaries = ()
        self.scalar_summaries_tf = ()
        self.histogram_summaries = ()
        self.histogram_summaries_tf = ()
        self.model = None
        self.summary_op = None
        self.summary = None
        self._loss = None
        self.train_op = None
        self._trainable_vars = None

    def get_params(self):
        return {attr_name: getattr(self, attr_name) for attr_name in self.PARAMETERS}

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

    @property
    def trainable_vars(self):
        if self._trainable_vars is None:
            self._trainable_vars = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        return self._trainable_vars

    def clip_action(self, action, **kwargs):
        return tf_v1.clip_by_value(action, self.action_space.low, self.action_space.high, **kwargs)

    def increment_schedulers(self):
        for scheduler in self.schedulers:
            scheduler.increment()

    def reshape_state(self, states):
        states = states.reshape(-1, *self.obs_shape)
        return states

    def action(self, sess, states, **kwargs):
        states = self.reshape_state(states)
        actions = self._action(sess, states, **kwargs)
        return actions

    def _action(self, sess, states, **kwargs):
        raise NotImplementedError

    def update(self, sess, states, actions, *args, **kwargs):
        return 0.0

    def hook_before_train(self, **kwargs):
        pass

    def hook_before_epoch(self, **kwargs):
        pass

    def hook_before_step(self, **kwargs):
        pass

    def hook_after_step(self, **kwargs):
        pass

    def hook_after_epoch(self, **kwargs):
        self.increment_schedulers()

    def hook_after_train(self, **kwargs):
        pass

    def save(self, file_path, verbose=True, **kwargs):
        policy_type = self.__class__.__name__
        if self.model is None:
            print(f"# {policy_type} has not defined any policy model!")
        else:
            self.model.save(file_path, save_format="tf")
            if verbose:
                print(f"# Saved {policy_type} to '{file_path}'!")

    def init_summaries(self, tag="", force=False):
        if self.summary_op is None or force:
            _summaries = []
            for summary_type in ("scalar", "histogram"):
                summary_func = getattr(tf_v1.summary, summary_type)
                for summary_attr in getattr(self, f"{summary_type}_summaries_tf"):
                    attr = getattr(self, summary_attr)
                    _summaries.append(summary_func(f"{tag}/{self.scope}/{summary_attr}", attr))
            if _summaries:
                self.summary_op = tf_v1.summary.merge(_summaries)
