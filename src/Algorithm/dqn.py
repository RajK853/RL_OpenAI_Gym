import numpy as np
from src.Layer import QNetwork
from .rl_algorithm import RLAlgorithm


class DQN(RLAlgorithm):

    def __init__(self, *, df, lr, layer_units=None, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.df = df
        self.lr = lr
        self.layer_units = layer_units or (50, 50, 50)
        self.q_net = QNetwork(self.obs_shape, self.action_size, layer_units=self.layer_units, scope="q_network")
        self.target_q = self.q_net
        self.mean_estimator_loss = 0.0
        self.summary_init_objects += (self.q_net,)
        self.scalar_summaries += ("eps", "epoch_length", "mean_estimator_loss")

    @property
    def eps(self):
        return self.policy.eps

    def action(self, sess, states, **kwargs):
        return self.policy.action(sess, states, estimator=self.q_net, **kwargs)

    def train_model(self, sess):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # Predict Q-values for next states
        current_qs = self.target_q.predict(sess, states)
        next_qs = self.target_q.predict(sess, next_states)
        q_targets = current_qs
        q_targets[np.arange(len(actions)), actions] = self.lr*(rewards + self.df*(1 - dones)*np.amax(next_qs, axis=-1))
        self.mean_estimator_loss = self.q_net.update(sess, states, q_targets)
        self.policy.update(sess, states=states, actions=actions)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.policy.hook_after_action(epoch=self.epoch, **kwargs)
            self.add_summaries()

    def hook_after_step(self, **kwargs):
        self.replay_buffer.add(self.transition)
        if self.training:
            self.train_model(self.sess)
