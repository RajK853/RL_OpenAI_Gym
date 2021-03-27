import tensorflow.compat.v1 as tf_v1

from .dqn import DQN
from src.Network.qnetwork import QNetwork


class DDQN(DQN):
    def __init__(self, *, tau, update_interval, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        self.target_q = QNetwork(input_shape=self.obs_shape, output_size=self.action_size,
                                 layers=self.layers, scope="target_q_network")
        self.target_q.init_weight_update_op(self.q_net)
        self.tau = tau
        self.update_interval = update_interval

    def get_q_target(self):
        q_targets = tf_v1.stop_gradient(super().get_q_target())
        return q_targets

    def hook_before_train(self, **kwargs):
        super().hook_before_train(**kwargs)
        self.target_q.update_weights(self.sess, tau=1.0)

    def hook_after_step(self, **kwargs):
        super().hook_after_step(**kwargs)
        if self.training and (not self.steps % self.update_interval):
            self.target_q.update_weights(self.sess, tau=self.tau)
