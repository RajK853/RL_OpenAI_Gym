import tensorflow.compat.v1 as tf_v1
from . import OnPolicyAlgorithm
from src.utils import get_scheduler
from src.Network.qnetwork import QNetwork
from src.Network.utils import get_clipped_train_op


l2 = tf_v1.keras.regularizers.l2

DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "LayerNormalization"},
    {"type": "Dense", "units": 256, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "Dense", "units": 1, "activation": None},
]


class Sarsa(OnPolicyAlgorithm):

    def __init__(self, lr_kwargs, gamma_kwargs, reward_scale=1.0, layers=None, num_trains=1, **kwargs):
        super(Sarsa, self).__init__(**kwargs)
        if layers is None:
            layers = DEFAULT_LAYERS
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.gamma_scheduler = get_scheduler(gamma_kwargs)
        self.reward_scale = reward_scale
        self.layers = layers
        self.num_trains = num_trains
        self.field_names = ("state", "action", "reward", "next_state", "done")
        self.q_net = QNetwork(input_shape=self.obs_shape, output_size=self.action_size, layers=layers, scope="q_network")
        self.target_q = self.q_net
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=[], name="lr_ph")
        self.gamma_ph = tf_v1.placeholder("float32", shape=[], name="gamma_ph")
        self.states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="states_ph")
        self.actions_ph = tf_v1.placeholder("int32", shape=[None], name="actions_ph")
        self.rewards_ph = tf_v1.placeholder("float32", shape=[None], name="rewards_ph")
        self.next_states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="next_states_ph")
        self.dones_ph = tf_v1.placeholder("float32", shape=[None], name="dones_ph")
        self.next_actions_ph = tf_v1.placeholder("int32", shape=[None], name="next_actions_ph")
        # Summary ops
        self.summary_init_objects += (self.q_net, )
        self.scalar_summaries += ("gamma", "lr")

    @property
    def gamma(self):
        return self.gamma_scheduler.value

    @property
    def lr(self):
        return self.lr_scheduler.value

    def action(self, state, **kwargs):
        return super().action(state, estimator=self.q_net, **kwargs)

    def get_q_target(self):
        """hot_encoded = tf_v1.one_hot(self.next_actions_ph, self.action_size)
        next_qs = self.target_q(self.next_states_ph)
        next_qs = tf_v1.reduce_sum(hot_encoded*next_qs, axis=-1)"""
        next_qs = self.target_q(self.next_states_ph)
        batch_size = tf_v1.shape(self.next_actions_ph)[0]
        indices = tf_v1.stack([tf_v1.range(batch_size), self.next_actions_ph], axis=-1)
        next_qs = tf_v1.gather_nd(next_qs, indices)
        q_targets = self.reward_scale*self.rewards_ph + self.gamma_ph*next_qs*(1.0 - self.dones_ph)
        return q_targets

    def init_q(self):
        """hot_encoded = tf_v1.one_hot(self.actions_ph, self.action_size)
        q_predictions = self.q_net(self.states_ph)
        q_predictions = tf_v1.reduce_sum(hot_encoded*q_predictions, axis=-1)"""
        q_targets = self.get_q_target()
        batch_size = tf_v1.shape(self.actions_ph)[0]
        indices = tf_v1.stack([tf_v1.range(batch_size), self.actions_ph], axis=-1)
        q_predictions = self.q_net(self.states_ph)
        q_targets = tf_v1.tensor_scatter_nd_update(q_predictions, indices, q_targets)
        q_loss = tf_v1.losses.mean_squared_error(labels=q_targets, predictions=q_predictions)
        # q_loss = tf_v1.squared_difference(q_targets, q_predictions)
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr_ph)
        train_op = get_clipped_train_op(q_loss, optimizer=optimizer, var_list=self.q_net.trainable_vars,
                                        clip_norm=self.clip_norm)
        self.q_net.setup_loss(q_loss, train_op)

    def hook_before_train(self, **kwargs):
        self.init_q()
        super().hook_before_train(**kwargs)

    def train(self):
        states, actions, rewards, next_states, dones = self.sample_trajectory()
        next_actions = self.action(next_states, deterministic=True)
        feed_dict = {self.states_ph: states,
                     self.actions_ph: actions,
                     self.rewards_ph: rewards,
                     self.next_states_ph: next_states,
                     self.dones_ph: dones,
                     self.next_actions_ph: next_actions,
                     self.gamma_ph: self.gamma,
                     self.lr_ph: self.lr}
        for _ in range(self.num_trains):
            self.q_net.update(self.sess, feed_dict)

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training:
            state, action, reward, next_state, done = self.transition
            self.add_transition(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.gamma_scheduler.increment()
            self.lr_scheduler.increment()
