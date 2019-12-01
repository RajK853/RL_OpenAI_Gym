import numpy as np
from .RLAlgorithm import RLAlgorithm
from src.Layer import QNetwork


class DQN(RLAlgorithm):

    def __init__(self, *, df, lr, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.df = df
        self.lr = lr
        self.local_estimator = QNetwork(self.observation_space.shape, self.action_size, scope="local_network")
        self.target_estimator = self.local_estimator

    def action(self, sess, states, **kwargs):
        return self.policy.action(sess, states, estimator=self.local_estimator, **kwargs)

    def train_model(self, sess):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        # Predict Q-values for next states
        q_sa_t1 = self.target_estimator.predict(sess, next_states)
        # Empty numpy array for q_targets
        q_targets = np.zeros(len(actions))
        for t, (future_q, reward, done) in enumerate(zip(q_sa_t1, rewards, dones)):
            if done:
                q_targets[t] = self.lr * reward
            else:
                q_targets[t] = self.lr * (reward + self.df * np.amax(future_q))
        # TODO: Change function name? Train_op? Compute_loss?
        batch_avg_loss = self.local_estimator.update(sess, states, q_targets, actions)
        policy_loss = self.policy.update(sess, states=states, actions=actions)
        return batch_avg_loss, policy_loss

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        self.policy.hook_after_action(epoch=self.epoch, **kwargs)

    def hook_after_step(self, **kwargs):
        if self.training:
            self.replay_buffer.add(self.transition)
            estimator_loss, policy_loss = self.train_model(self.sess)
            self._estimator_losses.append(estimator_loss)
            self._policy_losses.append(policy_loss)
