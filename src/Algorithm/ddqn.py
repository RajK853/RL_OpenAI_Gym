from .dqn import DQN
from src.Layer import QNetwork


class DDQN(DQN):
    def __init__(self, *, tau, update_interval, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        self.target_q = QNetwork(self.obs_shape, self.action_size, layer_units=self.layer_units, scope="target_network")
        self.target_q.init_weight_update_op(self.q_net)
        self.tau = tau
        self.update_interval = update_interval
        self.summary_init_objects += (self.target_q, )

    def hook_before_train(self, **kwargs):
        super().hook_before_train(**kwargs)
        self.target_q.update_weights(self.sess, tau=1.0)

    def hook_after_step(self, **kwargs):
        super(DDQN, self).hook_after_step(**kwargs)
        if self.training and (not self.steps % self.update_interval):
            self.target_q.update_weights(self.sess, tau=self.tau)
