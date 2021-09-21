import numpy as np
import tensorflow.compat.v1 as tf_v1

from .reinforce import Reinforce
from src.registry import registry
from src.utils import get_scheduler
from src.Network.qnetwork import QNetwork
from src.Network.utils import get_clipped_train_op

DEFAULT_KWARGS = {
    "lr_kwargs": {
        "type": "ConstantScheduler",
        "value": 0.0001,
    },
}


@registry.algorithm.register("a2c")
class A2C(Reinforce):
    PARAMETERS = Reinforce.PARAMETERS.union({"lr_kwargs"})

    def __init__(self, *, lr_kwargs=DEFAULT_KWARGS["lr_kwargs"], **kwargs):
        super(A2C, self).__init__(**kwargs)
        self.critic = QNetwork(input_shapes=[self.obs_shape], output_size=1, layers=self._layers, 
            preprocessors=self.preprocessors, scope="critic")
        self.lr_kwargs = lr_kwargs
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.schedulers += (self.lr_scheduler, )
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=(), name="lr_ph")
        self.summary_init_objects += (self.critic, )
        self.scalar_summaries += ("lr", )

    @property
    def layers(self):
        return self.critic.layers

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
            self.states_ph: states,               # TODO: Remove default placeholders and define it in the algo class instead?
            self.actions_ph: actions,
            self.targets_ph: advantage
        }
        self.policy.update(self.sess, feed_dict=feed_dict)

    def train_critic(self, states, targets):
        feed_dict = {
            self.critic.input: states,
            self.targets_ph: targets,
            self.lr_ph: self.lr
        }
        self.critic.update(self.sess, feed_dict=feed_dict)

    def train(self):
        states, actions, rewards = self.sample_trajectory()
        discounted_return = self.compute_discounted_return(rewards)
        discounted_return = np.expand_dims(discounted_return, axis=-1)
        values = self.critic.predict(states)                   # TODO: Directly connect its tf graph to the critic loss func 
        advantage = discounted_return - values
        for _ in range(self.num_gradient_steps):
            self.train_critic(states, discounted_return)
            self.train_actor(states, actions, advantage)

    def hook_before_train(self, **kwargs):
        self.init_critic()
        super().hook_before_train(**kwargs)
