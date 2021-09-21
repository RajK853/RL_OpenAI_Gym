import numpy as np
import tensorflow.compat.v1 as tf_v1

from .off_policy import OffPolicyAlgorithm
from src.Network.qnetwork import QNetwork
from src.registry import registry
from src.Network.utils import get_clipped_train_op
from src.utils import get_scheduler


DEFAULT_KWARGS = {
    "gamma_kwargs": {
        "type": "ConstantScheduler",
        "value": 0.99,
    },
    "q_lr_kwargs": {
        "type": "ConstantScheduler",
        "value": 0.0003,
    },
    "alpha_lr_kwargs": {
        "type": "ConstantScheduler",
        "value": 0.0003,
    },
}


@registry.algorithm.register("sac")
class SAC(OffPolicyAlgorithm):
    VALID_POLICIES = {"GaussianPolicy"}
    PARAMETERS = OffPolicyAlgorithm.PARAMETERS.union({
        "reward_scale", "gamma_kwargs", "alpha_lr_kwargs", "q_lr_kwargs", "tau", "update_interval", "num_q_nets",
        "auto_ent", "target_entropy", "init_log_alpha"
    })

    def __init__(self, *, reward_scale=1.0, gamma_kwargs=DEFAULT_KWARGS["gamma_kwargs"], alpha_lr_kwargs=DEFAULT_KWARGS["alpha_lr_kwargs"], 
        q_lr_kwargs=DEFAULT_KWARGS["q_lr_kwargs"], tau=5e-3, update_interval=1, num_q_nets=2, auto_ent=True, target_entropy="auto", 
        init_log_alpha=0.0, **kwargs):
        super(SAC, self).__init__(**kwargs)
        assert num_q_nets > 1, f"Minimum number of Q network is 2 but given '{num_q_nets}'"
        self.reward_scale = reward_scale
        self.q_lr_kwargs = q_lr_kwargs
        self.gamma_kwargs = gamma_kwargs
        self.alpha_lr_kwargs = alpha_lr_kwargs
        self.q_lr_scheduler = get_scheduler(q_lr_kwargs)
        self.gamma_scheduler = get_scheduler(gamma_kwargs)
        self.alpha_lr_scheduler = get_scheduler(alpha_lr_kwargs)
        self.schedulers += (self.q_lr_scheduler, self.gamma_scheduler, self.alpha_lr_scheduler)
        self.tau = tau
        self.update_interval = update_interval
        self.num_q_nets = num_q_nets
        self.auto_ent = auto_ent
        self.init_log_alpha = init_log_alpha
        if self.auto_ent:
            assert target_entropy == "auto" or isinstance(target_entropy, (int, float))
            self.target_entropy = -float(self.action_size) if target_entropy == "auto" else target_entropy
            self.log_alpha_tf = tf_v1.get_variable('log_alpha', dtype="float32", initializer=float(init_log_alpha))
        else:
            self.target_entropy = None
            self.log_alpha_tf = tf_v1.constant(init_log_alpha, dtype="float32")
        self.alpha_tf = tf_v1.exp(self.log_alpha_tf)
        self.log_alpha = 0.0
        self.alpha_loss_tf = None
        self.alpha_train_op = None
        self.critics = []
        self.targets = []
        # Placeholders
        self.q_lr_ph = tf_v1.placeholder("float32", shape=[], name="q_lr_ph")
        self.alpha_lr_ph = tf_v1.placeholder("float32", shape=[], name="alpha_lr_ph")
        self.gamma_ph = tf_v1.placeholder("float32", shape=[], name="gamma_ph")
        self.states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="states_ph")
        self.actions_ph = tf_v1.placeholder("float32", shape=[None, self.action_size], name="actions_ph")
        self.rewards_ph = tf_v1.placeholder("float32", shape=[None], name="rewards_ph")
        self.next_states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="next_states_ph")
        self.dones_ph = tf_v1.placeholder("float32", shape=[None], name="dones_ph")
        # Summary parameters
        self.alpha_loss = None
        self.scalar_summaries += ("alpha_loss", "log_alpha", "alpha", "q_lr", "alpha_lr")

    @property
    def gamma(self):
        return self.gamma_scheduler.value

    @property
    def q_lr(self):
        return self.q_lr_scheduler.value

    @property
    def alpha_lr(self):
        return self.alpha_lr_scheduler.value

    @property
    def alpha(self):
        return np.exp(self.log_alpha)

    @property
    def layers(self):
        return self.critics[0].layers

    def init_critics(self):
        q_nets = []
        target_q_nets = []
        q_net_kwargs = {
            "input_shapes": [self.obs_shape, self.action_shape],
            "output_size": 1,
            "layers": self._layers,
            "preprocessors": self.preprocessors}
        for i in range(self.num_q_nets):
            q_net = QNetwork(**q_net_kwargs, scope=f"critic_{i}")
            target_q_net = QNetwork(**q_net_kwargs, scope=f"targets_{i}")
            target_q_net.init_weight_update_op(q_net)
            q_nets.append(q_net)
            target_q_nets.append(target_q_net)
        self.critics = tuple(q_nets)
        self.targets = tuple(target_q_nets)
        self.init_critics_loss()
        self.summary_init_objects += self.critics

    def get_q_targets(self):
        next_actions = self.policy.model(self.next_states_ph)
        next_q_values = tuple(Q([self.next_states_ph, next_actions]) for Q in self.targets)
        next_min_q = tf_v1.reduce_min(next_q_values, axis=0)
        next_log_actions = self.policy.log_prob(self.next_states_ph, next_actions)
        next_qs = next_min_q - self.alpha_tf * next_log_actions
        q_targets = self.reward_scale * self.rewards_ph + self.gamma_ph * (1 - self.dones_ph) * next_qs
        return q_targets

    def init_critics_loss(self):
        q_targets = tf_v1.stop_gradient(self.get_q_targets())
        for critic in self.critics:
            q_predictions = critic([self.states_ph, self.actions_ph])
            q_loss = tf_v1.reduce_mean(tf_v1.losses.huber_loss(labels=q_targets, predictions=q_predictions))
            # q_loss = tf_v1.losses.mean_squared_error(labels=q_targets, predictions=q_predictions, weights=0.5)
            optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.q_lr_ph)
            train_op = get_clipped_train_op(q_loss, optimizer=optimizer, var_list=critic.trainable_vars,
                                            clip_norm=self.clip_norm)
            critic.setup_loss(q_loss, train_op)

    def init_actor(self):
        actions = self.policy.model(self.states_ph)
        q_values = tuple(Q([self.states_ph, actions]) for Q in self.critics)
        min_q = tf_v1.reduce_min(q_values, axis=0)
        log_actions = self.policy.log_prob(self.states_ph, actions)
        kl_loss = min_q - self.alpha_tf*log_actions
        policy_loss = -tf_v1.reduce_mean(kl_loss)
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.policy.lr_ph, name="policy_optimizer")
        actor_train_op = get_clipped_train_op(policy_loss, optimizer, var_list=self.policy.trainable_vars, 
            clip_norm=self.policy.clip_norm)
        self.policy.setup_loss(policy_loss, actor_train_op)

        if self.auto_ent:
            alpha_loss = -self.alpha_tf*tf_v1.stop_gradient(log_actions + self.target_entropy)
            self.alpha_loss_tf = tf_v1.reduce_mean(alpha_loss)
            optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.alpha_lr_ph, name="alpha_optimizer")
            self.alpha_train_op = optimizer.minimize(self.alpha_loss_tf, var_list=[self.log_alpha_tf])

    def update_alpha(self, feed_dict):
        if self.auto_ent:
            train_ops = [self.log_alpha_tf, self.alpha_loss_tf, self.alpha_train_op]
            self.log_alpha, self.alpha_loss, *_ = self.sess.run(train_ops, feed_dict=feed_dict)
        else:
            self.log_alpha = self.sess.run(self.log_alpha_tf)

    def update_target_weights(self, tau):
        for Q in self.targets:
            Q.update_weights(self.sess, tau)

    def train(self):
        for i in range(self.num_gradient_steps):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            feed_dict = {self.policy.model.input: states,  # This is required for the (log_actions, mu, std) summaries
                         self.states_ph: states,
                         self.actions_ph: actions,
                         self.rewards_ph: rewards,
                         self.next_states_ph: next_states,
                         self.dones_ph: dones,
                         self.gamma_ph: self.gamma,
                         self.q_lr_ph: self.q_lr}
            for critic in self.critics:
                critic.update(self.sess, feed_dict)

            feed_dict = {self.policy.model.input: states,
                         self.states_ph: states,
                         self.alpha_lr_ph: self.alpha_lr}
            self.policy.update(self.sess, feed_dict)
            self.update_alpha(feed_dict)

            if (i % self.update_interval) == 0:
                self.update_target_weights(tau=self.tau)

    def action(self, states, deterministic=False, **kwargs):
        action = self.policy.action(self.sess, states, deterministic=deterministic, **kwargs)
        return action

    def hook_before_train(self, **kwargs):
        self.init_critics()
        self.init_actor()
        super().hook_before_train(**kwargs)  # Variables initialized here
        self.update_target_weights(tau=1.0)
