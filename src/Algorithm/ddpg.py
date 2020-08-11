import numpy as np
import tensorflow.compat.v1 as tf_v1
from .rl_algorithm import RLAlgorithm
from src.Layer import QNetwork


class DDPG(RLAlgorithm):
    VALID_POLICIES = ["ContinuousPolicy"]

    def __init__(self, *, lr, gamma, tau, update_interval, layer_units, **kwargs):
        super(DDPG, self).__init__(**kwargs)
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.update_interval = update_interval
        # Build networks
        self.target_actor = self.policy.build_network(scope=f"{self.policy.scope}_target")
        self.target_actor.init_weight_update_op(self.policy.network)
        critic_input_shape = (self.observation_size+self.action_size, )
        self.critic = QNetwork(critic_input_shape, output_size=1, layer_units=layer_units, scope="critic")
        self.target_critic = QNetwork(critic_input_shape, output_size=1, layer_units=layer_units, scope="critic_target")
        self.target_critic.init_weight_update_op(self.critic)
        # Summary parameters
        self.summary_init_objects += (self.critic, self.target_critic)
        self.init_actor_loss()

    def init_actor_loss(self):
        loss = -self.critic(tf_v1.concat([self.policy.network.input, self.policy.network.output], axis=-1))
        self.policy.set_loss(loss)

    def train_actor(self, sess, states):
        feed_dict = {self.policy.network.input: states}
        self.policy.update(sess, feed_dict=feed_dict)

    def train_critic(self, sess, states, actions, rewards, next_states, dones):
        # Predict Q-values for next states
        next_actions = self.action(sess, next_states)
        next_qs = self.target_critic.predict(sess, np.concatenate([next_states, next_actions], axis=-1))
        q_targets = self.lr * (rewards + self.gamma * (1 - dones) * next_qs.squeeze())
        q_targets = q_targets.reshape((*q_targets.shape, 1))
        self.critic.update(sess, np.concatenate([states, actions], axis=-1), q_targets)

    def train(self, sess):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        self.train_critic(sess, states, actions, rewards, next_states, dones)
        self.train_actor(sess, states)

    def hook_before_train(self, **kwargs):
        super().hook_before_train(**kwargs)
        self.target_actor.update_weights(self.sess, tau=1.0)
        self.target_critic.update_weights(self.sess, tau=1.0)

    def hook_after_step(self, **kwargs):
        super(DDPG, self).hook_after_step(**kwargs)
        if self.training:
            self.replay_buffer.add(self.transition)
            self.train(self.sess)
            if not self.steps % self.update_interval:
                self.target_actor.update_weights(self.sess, tau=self.tau)
                self.target_critic.update_weights(self.sess, tau=self.tau)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.policy.hook_after_action(epoch=self.epoch, **kwargs)
            self.add_summaries()
