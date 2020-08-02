import numpy as np
from src.Layer import QNetwork
from .dqn import DQN


class Sarsa(DQN):

    def __init__(self, **kwargs):
        super(Sarsa, self).__init__(**kwargs)
        self.trajectory = {}

    def add_transition(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.trajectory.keys():
                self.trajectory[key] = []
            self.trajectory[key].append(value)

    def sample_trajectory(self):
        return {key: np.array(value) for key, value in self.trajectory.items()}

    def train(self, sess):
        trajectory = self.sample_trajectory()
        states = trajectory["state"]
        actions = trajectory["action"]
        rewards = trajectory["reward"]
        next_states = trajectory["next_state"]
        dones = trajectory["done"]
        # Predict Q-values for next states
        current_qs = self.target_q.predict(sess, states)
        next_qs = self.target_q.predict(sess, next_states)
        next_actions = self.action(sess, next_states)
        next_qs = next_qs[np.arange(len(next_actions)), next_actions]
        q_targets = current_qs
        q_targets[np.arange(len(actions)), actions] = self.lr*(rewards + self.gamma*(1 - dones)*next_qs)
        self.mean_estimator_loss = self.q_net.update(sess, states, q_targets)

    def hook_after_step(self, **kwargs):
        if self.training:
            state, action, reward, next_state, done = self.transition
            self.add_transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
            self.train(self.sess)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        if self.training:
            self.trajectory.clear()
