import numpy as np
import tensorflow.compat.v1 as tf_v1

from . import Reinforce
from src.utils import get_scheduler
from src.Network.qnetwork import QNetwork
from src.Network.utils import get_clipped_train_op


class ActorCritic(Reinforce):

    def __init__(self, *, lr_kwargs, baseline_scale=0.1, layers=None, **kwargs):
        super(ActorCritic, self).__init__(**kwargs)
        self.baseline_scale = baseline_scale
        self.critic = QNetwork(input_shape=self.obs_shape, output_size=1, layers=layers, scope="critic")
        self.lr_scheduler = get_scheduler(lr_kwargs)
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=(), name="lr_ph")
        self.targets_ph = tf_v1.placeholder("float32", shape=(None, 1), name="targets_ph")
        self.summary_init_objects += (self.critic, )
        self.scalar_summaries += ("lr", )

    @property
    def lr(self):
        return self.lr_scheduler.value

    def init_critic(self):
        loss = tf_v1.losses.mean_squared_error(labels=self.targets_ph, predictions=self.critic.output)
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr_ph)
        train_op = get_clipped_train_op(loss, optimizer=optimizer, var_list=self.critic.trainable_vars,
                                        clip_norm=self.clip_norm)
        self.critic.setup_loss(loss, train_op)

    def train_actor(self, states, actions, advantage):
        feed_dict = {
            self.policy.model.input: states,
            self.policy.actions_ph: actions,
            self.policy.targets_ph: advantage}
        self.policy.update(self.sess, feed_dict=feed_dict)

    def train_critic(self, states, targets):
        feed_dict = {
            self.critic.input: states,
            self.targets_ph: targets,
            self.lr_ph: self.lr}
        self.critic.update(self.sess, feed_dict=feed_dict)

    def train(self):
        states, actions, rewards = self.sample_trajectory()
        discounted_return = self.compute_discounted_return(rewards)
        # discounted_return = standardize_array(discounted_return)
        discounted_return = np.expand_dims(discounted_return, axis=-1)
        values = self.critic.predict(states)
        advantage = discounted_return - values
        for _ in range(self.num_train):
            self.train_critic(states, advantage)
            self.train_actor(states, actions, advantage)

    def hook_before_train(self, **kwargs):
        self.init_critic()
        super().hook_before_train(**kwargs)

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        self.lr_scheduler.increment()
