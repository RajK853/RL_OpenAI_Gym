import numpy as np
import tensorflow.compat.v1 as tf_v1
from . import OffPolicyAlgorithm
from src.Network import NeuralNetwork, QNetwork
from src.Network.utils import get_clipped_train_op
from src.utils import get_scheduler


class DDPG(OffPolicyAlgorithm):
    VALID_POLICIES = ["ContinuousPolicy"]

    def __init__(self, *, tau, update_interval, lr_kwargs, gamma_kwargs, sigma_kwargs, reward_scale=1.0, num_q_nets=2,
                 layers=None, random_type="normal", **kwargs):
        super(DDPG, self).__init__(**kwargs)
        self.tau = tau
        self.reward_scale = reward_scale
        self.update_interval = update_interval
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.gamma_scheduler = get_scheduler(gamma_kwargs)
        self.sigma_scheduler = get_scheduler(sigma_kwargs)
        self.random_callable = self.get_random_function(random_type)
        # Build networks
        self.target_actor = NeuralNetwork(scope=f"{self.policy.scope}_target", input_shape=self.policy.obs_shape,
                                          layers=self.policy.layers)
        self.target_actor.init_weight_update_op(self.policy.model)
        self.layers = layers
        self.num_q_nets = num_q_nets
        self.critics = []
        self.targets = []
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

    def init_critics(self):
        q_nets = []
        target_q_nets = []
        input_shape = (self.observation_size + self.action_size, )
        for i in range(self.num_q_nets):
            q_net = QNetwork(input_shape=input_shape, output_size=1, layers=self.layers, scope=f"critic_{i}")
            target_q_net = QNetwork(input_shape=input_shape, output_size=1, layers=self.layers, scope=f"targets_{i}")
            target_q_net.init_weight_update_op(q_net)
            q_nets.append(q_net)
            target_q_nets.append(target_q_net)
        self.critics = tuple(q_nets)
        self.targets = tuple(target_q_nets)
        self.init_critics_loss()
        self.summary_init_objects += self.critics

    def get_q_targets(self):
        next_actions = self.target_actor(self.next_states_ph)
        next_states_actions = tf_v1.concat([self.next_states_ph, next_actions], axis=-1)
        next_qs = tuple(Q(next_states_actions) for Q in self.targets)
        min_next_qs = tf_v1.reduce_min(next_qs, axis=0)
        q_targets = self.reward_scale * self.rewards_ph + self.gamma_ph * min_next_qs * (1.0 - self.dones_ph)
        return q_targets

    def init_critics_loss(self):
        q_targets = tf_v1.stop_gradient(self.get_q_targets())
        states_actions = tf_v1.concat([self.states_ph, self.actions_ph], axis=-1)
        for critic in self.critics:
            q_predictions = critic(states_actions)
            q_loss = tf_v1.losses.mean_squared_error(labels=q_targets, predictions=q_predictions, weights=0.5)
            optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr_ph)
            train_op = get_clipped_train_op(q_loss, optimizer=optimizer, var_list=critic.trainable_vars,
                                            clip_norm=self.clip_norm)
            critic.setup_loss(q_loss, train_op)

    def init_actor_loss(self):
        actions = self.policy.model(self.states_ph)
        state_action = tf_v1.concat([self.states_ph, actions], axis=-1)
        q_values = tuple(Q(state_action) for Q in self.critics)
        min_q = tf_v1.reduce_mean(q_values, axis=0)
        loss = -tf_v1.reduce_mean(min_q)
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.policy.lr_ph)
        train_op = get_clipped_train_op(loss, optimizer, var_list=self.policy.trainable_vars,
                                        clip_norm=self.policy.clip_norm)
        self.policy.setup_loss(loss, train_op)

    def get_random_function(self, random_type):
        if random_type == "uniform":
            random_callable = lambda action: self.sigma*np.random.uniform(-1, 1, size=action.shape)
        elif random_type == "normal":
            random_callable = lambda action: np.random.normal(loc=0.0, scale=self.sigma, size=action.shape)
        else:
            raise NotImplementedError(f"Invalid random type! Received '{random_type}'.")
        return random_callable

    def action(self, states, deterministic=False, **kwargs):
        action = self.policy.action(self.sess, states, **kwargs)
        if not deterministic:
            action += self.random_callable(action)
            action = np.clip(action, -1, 1)
        return action

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size * self.num_q_nets)
        for i, critic in enumerate(self.critics):
            start_index = i * self.batch_size
            end_index = (i + 1) * self.batch_size
            feed_dict = {self.states_ph: states[start_index:end_index],
                         self.actions_ph: actions[start_index:end_index],
                         self.rewards_ph: rewards[start_index:end_index],
                         self.next_states_ph: next_states[start_index:end_index],
                         self.dones_ph: dones[start_index:end_index],
                         self.gamma_ph: self.gamma,
                         self.lr_ph: self.lr}
            critic.update(self.sess, feed_dict)
        # Train actor
        indexes = np.random.choice(self.batch_size * self.num_q_nets, size=self.batch_size)
        feed_dict = {self.states_ph: states[indexes, :]}
        self.policy.update(self.sess, feed_dict)

    def update_target_weights(self, tau):
        self.target_actor.update_weights(self.sess, tau=tau)
        for Q in self.targets:
            Q.update_weights(self.sess, tau)

    def hook_before_train(self, **kwargs):
        # Init actor and critic losses
        self.init_critics()
        self.init_actor_loss()
        super().hook_before_train(**kwargs)
        self.update_target_weights(tau=1.0)

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training:
            self.replay_buffer.add(self.transition)
            self.train()
            if not self.steps % self.update_interval:
                self.update_target_weights(tau=self.tau)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.add_summaries(self.epoch)
            self.sigma_scheduler.increment()
            self.gamma_scheduler.increment()
            self.lr_scheduler.increment()
