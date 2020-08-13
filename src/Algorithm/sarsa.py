import numpy as np
from . import OnPolicyAlgorithm
from .dqn import DQN


class Sarsa(DQN, OnPolicyAlgorithm):

    def __init__(self, **kwargs):
        super(Sarsa, self).__init__(**kwargs)
        self.field_names = ("state", "action", "reward", "next_state", "done")
        self.trajectory = {}

    def train(self):
        states, actions, rewards, next_states, dones = self.sample_trajectory()
        # Predict Q-values for next states
        current_qs = self.target_q.predict(self.sess, states)
        next_qs = self.target_q.predict(self.sess, next_states)
        next_actions = self.action(next_states)
        next_qs = next_qs[np.arange(len(next_actions)), next_actions]
        q_targets = current_qs
        q_targets[np.arange(len(actions)), actions] = self.lr*(rewards + self.gamma*(1 - dones)*next_qs)
        self.mean_estimator_loss = self.q_net.update(self.sess, states, q_targets)

    def hook_after_step(self, **kwargs):
        OnPolicyAlgorithm.hook_after_step(self, **kwargs)
        if self.training:
            state, action, reward, next_state, done = self.transition
            self.add_transition(state=state, action=action, reward=reward, next_state=next_state, done=done)

    def hook_after_epoch(self, **kwargs):
        OnPolicyAlgorithm.hook_after_epoch(self, **kwargs)
        if self.training:
            self.policy.hook_after_epoch(epoch=self.epoch)
            self.train()
            self.add_summaries()
            self.clear_trajectory()
