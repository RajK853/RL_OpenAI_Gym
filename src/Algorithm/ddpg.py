import numpy as np
import tensorflow.compat.v1 as tf_v1

from .off_policy import OffPolicyAlgorithm
from src.registry import registry
from src.utils import get_scheduler
from src.Network import NeuralNetwork, QNetwork
from src.Network.utils import get_clipped_train_op

DEFAULT_KWARGS = {
    "sigma_kwargs": {
        "type": "ConstantScheduler",
        "value": 1.0,
    },
    "gamma_kwargs": {
        "type": "ConstantScheduler",
        "value": 0.99,
    },
    "lr_kwargs": {
        "type": "ConnstantScheduler",
        "value": 0.0001,
    },
}

@registry.algorithm.register("ddpg")
class DDPG(OffPolicyAlgorithm):
    VALID_POLICIES = ["ContinuousPolicy"]
    PARAMETERS = OffPolicyAlgorithm.PARAMETERS.union({
        "tau", "update_interval", "lr_kwargs", "gamma_kwargs", "sigma_kwargs", "reward_scale"
    })

    def __init__(self, *, tau=0.003, update_interval=10, lr_kwargs=DEFAULT_KWARGS["lr_kwargs"], gamma_kwargs=DEFAULT_KWARGS["gamma_kwargs"], 
        sigma_kwargs=DEFAULT_KWARGS["sigma_kwargs"], reward_scale=1.0, **kwargs):
        super(DDPG, self).__init__(**kwargs)
        self.tau = tau
        self.reward_scale = reward_scale
        self.update_interval = update_interval
        self.lr_kwargs = lr_kwargs
        self.gamma_kwargs = gamma_kwargs
        self.sigma_kwargs = sigma_kwargs
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.gamma_scheduler = get_scheduler(gamma_kwargs)
        self.sigma_scheduler = get_scheduler(sigma_kwargs)
        self.schedulers += (self.lr_scheduler, self.gamma_scheduler, self.sigma_scheduler)
        self.critic = None
        self.target = None
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=[], name="lr_ph")
        self.gamma_ph = tf_v1.placeholder("float32", shape=[], name="gamma_ph")
        self.states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="states_ph")
        self.actions_ph = tf_v1.placeholder("float32", shape=[None, self.action_size], name="actions_ph")
        self.rewards_ph = tf_v1.placeholder("float32", shape=[None], name="rewards_ph")
        self.next_states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="next_states_ph")
        self.dones_ph = tf_v1.placeholder("float32", shape=[None], name="dones_ph")
        # Summary parameters
        self.scalar_summaries += ("gamma", "sigma", "lr")

    @property
    def gamma(self):
        return self.gamma_scheduler.value

    @property
    def lr(self):
        return self.lr_scheduler.value

    @property
    def sigma(self):
        return self.sigma_scheduler.value

    @property
    def layers(self):
        return self.critic.layers

    def init_critics(self):
        q_nets = []
        target_q_nets = []
        q_net_kwargs = {
            "input_shapes": [self.obs_shape, self.action_shape],
            "output_size": 1,
            "layers": self._layers,
            "preprocessors": self.preprocessors}
        self.critic = QNetwork(**q_net_kwargs, scope="critic")
        self.target = QNetwork(**q_net_kwargs, scope="target_critic")
        self.target.init_weight_update_op(self.critic)
        self.init_critics_loss()
        self.summary_init_objects += (self.critic, )

    def get_q_targets(self):
        next_actions = self.policy.model(self.next_states_ph)
        next_qs = self.target([self.next_states_ph, next_actions])
        q_targets = self.reward_scale * self.rewards_ph + self.gamma_ph * next_qs * (1.0 - self.dones_ph)
        return q_targets

    def init_critics_loss(self):
        q_targets = tf_v1.stop_gradient(self.get_q_targets())
        q_predictions = self.critic([self.states_ph, self.actions_ph])
        # q_loss = tf_v1.losses.mean_squared_error(labels=q_targets, predictions=q_predictions, weights=0.5)
        q_loss = tf_v1.reduce_mean(tf_v1.losses.huber_loss(labels=q_targets, predictions=q_predictions))
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr_ph)
        train_op = get_clipped_train_op(q_loss, optimizer=optimizer, var_list=self.critic.trainable_vars,
                                        clip_norm=self.clip_norm)
        self.critic.setup_loss(q_loss, train_op)

    def init_actor_loss(self):
        actions = self.policy.model(self.states_ph)
        q_value = self.critic([self.states_ph, actions])
        loss = -tf_v1.reduce_mean(q_value)
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.policy.lr_ph)
        train_op = get_clipped_train_op(loss, optimizer, var_list=self.policy.trainable_vars,
                                        clip_norm=self.policy.clip_norm)
        self.policy.setup_loss(loss, train_op)

    def action(self, states, deterministic=False, **kwargs):
        action = self.policy.action(self.sess, states, **kwargs)
        if not deterministic:
            action += np.random.uniform(-self.sigma, self.sigma, size=action.shape)
            action = np.clip(action, -1, 1)
        return action

    def train(self):
        for i in range(self.num_gradient_steps):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            feed_dict = {self.states_ph: states,
                         self.actions_ph: actions,
                         self.rewards_ph: rewards,
                         self.next_states_ph: next_states,
                         self.dones_ph: dones,
                         self.gamma_ph: self.gamma,
                         self.lr_ph: self.lr}
            self.critic.update(self.sess, feed_dict)
            # Train actor
            feed_dict = {self.states_ph: states}
            self.policy.update(self.sess, feed_dict)

            if (i % self.update_interval) == 0:
                self.update_target_weights(tau=self.tau)

    def update_target_weights(self, tau):
        self.target.update_weights(self.sess, tau)

    def hook_before_train(self, **kwargs):
        # Init actor and critic losses
        self.init_critics()
        self.init_actor_loss()
        super().hook_before_train(**kwargs)
        self.update_target_weights(tau=1.0)
