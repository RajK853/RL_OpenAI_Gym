import tensorflow.compat.v1 as tf_v1

from .dqn import DQN
from src.Network.qnetwork import QNetwork


class DDQN(DQN):
    def __init__(self, *, tau=0.005, update_interval=10, **kwargs):
        super(DDQN, self).__init__(**kwargs)
        self.target_q = QNetwork(input_shapes=[self.obs_shape], output_size=self.action_size, layers=self.layers, 
            preprocessors=self.preprocessors, scope="target_q_network")
        self.target_q.init_weight_update_op(self.q_net)
        self.tau = tau
        self.update_interval = update_interval

    def get_q_target(self):
        q_targets = tf_v1.stop_gradient(super().get_q_target())
        return q_targets

    def hook_before_train(self, **kwargs):
        super().hook_before_train(**kwargs)
        self.target_q.update_weights(self.sess, tau=1.0)

    def train(self):
        for i in range(self.num_gradient_steps):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            feed_dict = {self.states_ph: states,
                         self.actions_ph: actions,
                         self.rewards_ph: rewards,
                         self.next_states_ph: next_states,
                         self.dones_ph: dones,
                         self.lr_ph: self.lr}
            self.q_net.update(self.sess, feed_dict)            
            if (i % self.update_interval) == 0:
                self.target_q.update_weights(self.sess, tau=self.tau)

