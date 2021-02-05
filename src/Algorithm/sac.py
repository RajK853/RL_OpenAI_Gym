import numpy as np
import tensorflow.compat.v1 as tf_v1
import tensorflow_probability as tfp

from src.Layer import QNetwork
from . import OffPolicyAlgorithm

tfp_dist = tfp.distributions


class SAC(OffPolicyAlgorithm):
    VALID_POLICIES = {"GaussianPolicy"}

    def __init__(self, *, gamma, reward_scale, tau, update_interval, num_q_nets=2, alpha_lr=1e-4, q_lr=1e-4,
                 auto_ent=True, target_entropy="auto", init_log_alpha=0.0, layer_units=None, layer_kwargs=None, **kwargs):
        super(SAC, self).__init__(**kwargs)
        assert num_q_nets > 1, f"Minimum number of Q network is 2 but given '{num_q_nets}'"
        self.reward_scale = reward_scale
        self.gamma = gamma
        self.tau = tau
        self.update_interval = update_interval
        self.num_q_nets = num_q_nets
        self.alpha_lr = alpha_lr
        self.q_lr = q_lr
        self.layer_units = layer_units or (50, 50, 50)
        self.layer_kwargs = layer_kwargs
        self.auto_ent = auto_ent
        if self.auto_ent:
            assert target_entropy == "auto" or isinstance(target_entropy, (int, float))
            self.target_entropy = -0.5*self.action_size if target_entropy == "auto" else target_entropy
            self.log_alpha_tf = tf_v1.get_variable('log_alpha', dtype=tf_v1.float32, initializer=np.float32(init_log_alpha))
        else:
            self.target_entropy = None
            self.log_alpha_tf = tf_v1.constant(np.float32(init_log_alpha), dtype=tf_v1.float32)
        self.alpha_tf = tf_v1.exp(self.log_alpha_tf)
        self.log_alpha = 0.0
        self.grad_clip = 1.0
        self.alpha_loss_tf = None
        self.alpha_train_op = None
        self.critics = None
        self.targets = None
        self.init_algo()
        # Summary parameters
        self.alpha_loss = None
        self.scalar_summaries += ("alpha_loss", "log_alpha", "alpha")
        self.summary_init_objects += self.critics + self.targets

    @property
    def alpha(self):
        return np.exp(self.log_alpha)

    def init_algo(self):
        self.init_critics()
        self.init_actor()

    def init_critics(self):
        q_nets = []
        target_q_nets = []
        input_shape = (self.observation_size + self.action_size, )
        q_net_kwargs = {"output_size": 1,
                        "layer_units": self.layer_units,
                        "lr": self.q_lr,
                        "weights": 0.5,
                        "layer_kwargs": self.layer_kwargs}
        for i in range(self.num_q_nets):
            q_net = QNetwork(input_shape, scope=f"critic_{i}", **q_net_kwargs)
            target_q_net = QNetwork(input_shape, scope=f"critic_target_{i}", **q_net_kwargs)
            target_q_net.init_weight_update_op(q_net)
            q_nets.append(q_net)
            target_q_nets.append(target_q_net)
        self.critics = tuple(q_nets)
        self.targets = tuple(target_q_nets)        

    def init_actor(self):
        states_ph = self.policy.input
        actions = self.policy.actions
        log_actions = self.policy.log_pi
        state_action = tf_v1.concat([states_ph, actions], axis=-1)
        q_values = tuple(Q(state_action) for Q in self.critics)
        min_q = tf_v1.reduce_min(q_values, axis=0)
        # print(log_actions, min_q)
        kl_loss = tf_v1.reduce_mean(self.alpha_tf*log_actions - min_q)
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.policy.lr, name="policy_optimizer")
        actor_train_op = optimizer.minimize(kl_loss, var_list=self.policy.trainable_vars)
        self.policy.set_loss(kl_loss, train_op=actor_train_op)

        if self.auto_ent:
            self.alpha_loss_tf = -tf_v1.reduce_mean(self.alpha_tf*(log_actions+self.target_entropy))
            optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.alpha_lr, name="alpha_optimizer")
            self.alpha_train_op = optimizer.minimize(self.alpha_loss_tf, var_list=[self.log_alpha_tf])

    def update_alpha(self, states):
        if self.auto_ent:
            feed_dict = {self.policy.input: states}
            train_ops = [self.log_alpha_tf, self.alpha_loss_tf, self.alpha_train_op]
            self.log_alpha, self.alpha_loss, *_ = self.sess.run(train_ops, feed_dict=feed_dict)
        else:
            self.log_alpha = self.sess.run(self.log_alpha_tf)

    def train_actor(self, states):
        self.policy.update(self.sess, states)

    def train_critic(self, states, actions, rewards, next_states, dones):
        next_actions = self.action(next_states)
        next_log_actions = self.policy.log_action(self.sess, next_states, next_actions)
        next_state_actions = np.concatenate([next_states, next_actions], axis=-1)
        next_q_values = tuple(Q.predict(self.sess, next_state_actions) for Q in self.targets)
        next_min_q = np.min(next_q_values, axis=0).squeeze()
        next_qs = next_min_q - self.alpha*next_log_actions
        # Q(s, a) = r(s, a) + gamma*V(s') if s is not a terminal state else r(s, a)
        q_targets = self.reward_scale*rewards + self.gamma * (1 - dones) * next_qs  # .squeeze()
        # print(rewards.shape, next_qs.shape, q_targets.shape)
        q_targets = q_targets.reshape((-1, 1))
        states_actions = np.concatenate([states, actions], axis=-1)
        for critic in self.critics:
            critic.update(self.sess, states_actions, q_targets)

    def update_target_weights(self, tau):
        for Q in self.targets:
            Q.update_weights(self.sess, tau)

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        self.train_critic(states, actions, rewards, next_states, dones)
        self.train_actor(states)
        self.update_alpha(states)

    def hook_before_train(self, **kwargs):
        super().hook_before_train(**kwargs)
        self.update_target_weights(tau=1.0)

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training:
            self.replay_buffer.add(self.transition)
            # if len(self.replay_buffer) >= self.batch_size:
            self.train()
            if self.steps % self.update_interval == 0:
                self.update_target_weights(tau=self.tau)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.policy.hook_after_epoch(**kwargs)
            self.add_summaries(self.epoch)
