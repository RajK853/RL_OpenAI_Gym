import tensorflow.compat.v1 as tf_v1

from . import OffPolicyAlgorithm
from src.utils import get_scheduler
from src.Network import QNetwork
from src.Network.utils import get_clipped_train_op


class DQN(OffPolicyAlgorithm):
    VALID_POLICIES = ["GreedyEpsilonPolicy"]

    def __init__(self, *, lr_kwargs, gamma_kwargs, reward_scale=1.0, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.gamma_scheduler = get_scheduler(gamma_kwargs)
        self.schedulers += (self.lr_scheduler, self.gamma_scheduler)
        self.reward_scale = reward_scale
        self.q_net = QNetwork(input_shapes=[self.obs_shape], output_size=self.action_size, layers=self.layers, 
            preprocessors=self.preprocessors, scope="q_network")
        self.target_q = self.q_net
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=[], name="lr_ph")
        self.states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="states_ph")
        self.actions_ph = tf_v1.placeholder("int32", shape=[None], name="actions_ph")
        self.rewards_ph = tf_v1.placeholder("float32", shape=[None], name="rewards_ph")
        self.next_states_ph = tf_v1.placeholder("float32", shape=[None, *self.obs_shape], name="next_states_ph")
        self.dones_ph = tf_v1.placeholder("float32", shape=[None], name="dones_ph")
        # Summary ops
        self.summary_init_objects += (self.q_net, )
        self.scalar_summaries += ("gamma", "lr")

    @property
    def gamma(self):
        return self.gamma_scheduler.value

    @property
    def lr(self):
        return self.lr_scheduler.value

    def get_q_target(self):
        next_qs = self.target_q(self.next_states_ph)
        max_next_qs = tf_v1.reduce_max(next_qs, axis=-1)
        q_targets = self.reward_scale*self.rewards_ph + self.gamma*max_next_qs*(1.0 - self.dones_ph)
        return q_targets

    def init_q(self):
        q_targets = self.get_q_target()
        batch_size = tf_v1.shape(self.actions_ph)[0]
        indices = tf_v1.stack([tf_v1.range(batch_size), self.actions_ph], axis=-1)
        q_predictions = self.q_net(self.states_ph)
        q_targets = tf_v1.tensor_scatter_nd_update(q_predictions, indices, q_targets)
        # q_predictions = tf_v1.gather_nd(q_predictions, indices)
        # q_loss = tf_v1.reduce_mean(tf_v1.losses.huber_loss(labels=q_targets, predictions=q_predictions))
        q_loss = tf_v1.losses.mean_squared_error(labels=q_targets, predictions=q_predictions)
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr_ph)
        train_op = get_clipped_train_op(q_loss, optimizer=optimizer, var_list=self.q_net.trainable_vars,
                                        clip_norm=self.clip_norm)
        self.q_net.setup_loss(q_loss, train_op)

    def hook_before_train(self, **kwargs):
        self.init_q()
        self.policy.set_model(self.q_net)
        super().hook_before_train(**kwargs)

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
