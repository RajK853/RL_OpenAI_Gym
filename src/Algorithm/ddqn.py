import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from .dqn import DQN
from src.Layer import QNetwork


class DDQN(DQN):

    def __init__(self, *, tau, update_interval, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        self.target_estimator = QNetwork(self.observation_space.shape, self.action_size, scope="target_network")
        local_vars = self.local_estimator.trainable_vars
        target_vars = self.target_estimator.trainable_vars
        self.tau_ph = tf_v1.placeholder(tf.float32, name="tau")
        self.update_ops = self.weight_update_ops(target_vars, local_vars)
        self.update_interval = update_interval
        self.tau = tau

    def weight_update_ops(self, target_vars, local_vars):
        update_ops = tf.group([tf_v1.assign(t_var, t_var + self.tau_ph * (l_var - t_var))
                               for t_var, l_var in zip(target_vars, local_vars)])
        return update_ops

    def update_weights(self, tau):
        self.sess.run(self.update_ops, feed_dict={self.tau_ph: tau})

    def hook_before_train(self, **kwargs):
        super().hook_before_train(**kwargs)
        self.update_weights(tau=1.0)

    def hook_after_step(self, **kwargs):
        super(DDQN, self).hook_after_step(**kwargs)
        if self.training and (not self.steps % self.update_interval):
            self.update_weights(self.tau)
