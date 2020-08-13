import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import tensorflow_probability as tfp

from src.Layer import NeuralNetwork
from .base_policy import BasePolicy

MIN_LOG_STD = -20
MAX_LOG_STD = 2

# TODO: Use tf.keras.Model for policy network


class GaussianPolicy(BasePolicy):

    def __init__(self, *, layer_units, alpha=1e-3, lr=1e-3, **kwargs):
        super(GaussianPolicy, self).__init__(env=kwargs.pop("env"))
        self.alpha = alpha
        self.lr = lr
        assert not self.discrete_action_space, "Action space for the Gaussian Policy must be continuous!"
        self.actions_ph = tf_v1.placeholder(tf.float32, shape=(None, *self.action_shape), name="action_ph")
        self.target_ph = tf_v1.placeholder(tf.float32, shape=(None,), name="target_ph")
        self.network = NeuralNetwork(scope=self.scope, input_shape=self.obs_shape, layer_units=layer_units,
                                     output_size=2*self.action_size, **kwargs)
        # Split mean and std from the final layer
        self.mu = self.network.output[:, :self.action_size]
        self.mu = tf_v1.clip_by_value(self.mu, self.action_space.low, self.action_space.high)
        self.log_std = self.network.output[:, self.action_size:]
        self.log_std = tf_v1.clip_by_value(self.log_std, MIN_LOG_STD, MAX_LOG_STD)
        self.norm_dist = tfp.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=tf_v1.exp(self.log_std))
        self.log_actions = self.norm_dist.log_prob(self.actions_ph)
        sample_action = self.norm_dist.sample()
        self.actions = tf.clip_by_value(sample_action, self.action_space.low, self.action_space.high)
        # Loss parameters
        self._loss = None
        self.train_op = None
        # Summary parameters
        self.scalar_summaries += ("mean_loss", )

    def init_default_loss(self):
        log_loss = self.log_actions * self.target_ph
        entropy_loss = -self.alpha * self.norm_dist.entropy()
        loss = log_loss + entropy_loss
        self.set_loss(loss=loss)

    def set_loss(self, loss, optimizer=None):
        self._loss = loss
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr) if optimizer is None else optimizer
        self.train_op = optimizer.minimize(self._loss, var_list=self.network.trainable_vars)

    @property
    def mean_loss(self):
        return tf.reduce_mean(self.loss)

    def _action(self, sess, states, **kwargs):
        actions = sess.run(self.actions, feed_dict={self.network.inputs_ph: states})
        return actions

    def log_action(self, sess, states, actions, **kwargs):
        log_actions = sess.run(self.log_actions, feed_dict={self.network.inputs_ph: states, self.actions_ph: actions})
        return log_actions

    def update(self, sess, states, actions, **kwargs):
        feed_dict = {self.network.inputs_ph: states, self.actions_ph: actions, self.target_ph: kwargs["targets"]}
        _,  loss, self.summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss
