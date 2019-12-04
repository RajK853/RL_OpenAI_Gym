import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from gym.spaces import Box

from src.Layer import NeuralNetwork
from .base_policy import BasePolicy


class GaussianPolicy(BasePolicy):

    def __init__(self, *, layer_units, scope="Gaussian_Policy", **kwargs):
        assert isinstance(self.action_space, Box), "Action space for the Gaussian Policy not continuous!"
        super(GaussianPolicy, self).__init__(**kwargs)
        self.actions_ph = tf_v1.placeholder(tf.float32, shape=[None, ], name="actions_ph")
        self.network = NeuralNetwork(scope, input_shape=self.obs_shape, layer_units=layer_units,
                                     output_size=2)
        # Split mean and std from the final layer
        mu = self.network.output[:, 0]
        # TODO: Adding 1e-5 to a tensor slows down the process
        std = tf.nn.relu(self.network.output[:, 1]) + 1e-5                       # Relu activation to keep std positive
        self.norm_dist = tf.contrib.distributions.Normal(mu, std)
        sample_actions = tf.math.reduce_mean(self.norm_dist.sample(3), axis=0)   # Mean of multiple samples
        self.actions_tf = tf.clip_by_value(sample_actions, self.action_space.low[0], self.action_space.high[0])
        # TODO: Code below not completed
        # Loss and train op
        # TODO: Loss function incomplete
        self.loss = -self.norm_dist.log_prob(self.actions_tf)
        # Add cross entropy cost to encourage exploration
        self.loss -= 0.1 * self.norm_dist.entropy()
        # Optimizer Parameters
        # Optimizer Parameters
        var_list = self.network.trainable_vars
        self.train_op = tf_v1.train.AdamOptimizer().minimize(self.loss, var_list=var_list)

    def action(self, sess, states, **kwargs):
        actions = sess.run(self.actions_tf, feed_dict={self.network.inputs_ph: states})
        return actions

    def update(self, sess, states, actions, **kwargs):
        feed_dict = {self.network.inputs_ph: states, self.actions_ph: actions}
        _,  loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss