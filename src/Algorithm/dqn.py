import numpy as np
from src.Layer import QNetwork
from . import OffPolicyAlgorithm


class DQN(OffPolicyAlgorithm):
    VALID_POLICIES = ["GreedyEpsilonPolicy"]

    def __init__(self, *, gamma=0.9, lr=0.99, layer_units=None, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.gamma = gamma
        self.lr = lr
        self.layer_units = layer_units or (50, 50, 50)
        self.q_net = QNetwork(self.obs_shape, self.action_size, layer_units=self.layer_units, scope="q_network")
        self.target_q = self.q_net
        self.mean_estimator_loss = 0.0
        self.summary_init_objects += (self.q_net,)
        self.scalar_summaries += ("eps", "mean_estimator_loss")

    @property
    def eps(self):
        return self.policy.eps

    def action(self, states, **kwargs):
        return super().action(states, estimator=self.q_net, **kwargs)

    def train(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # Predict Q-values for next states
        current_qs = self.target_q.predict(self.sess, states)
        next_qs = self.target_q.predict(self.sess, next_states)
        q_targets = current_qs
        q_targets[np.arange(len(actions)), actions] = self.lr*(rewards + self.gamma*(1 - dones)*np.amax(next_qs, axis=-1))
        self.mean_estimator_loss = self.q_net.update(self.sess, states, q_targets)

    def hook_after_step(self, **kwargs):
        if self.training:
            self.replay_buffer.add(self.transition)
            self.train()

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.policy.hook_after_epoch(epoch=self.epoch, **kwargs)
            self.add_summaries()
