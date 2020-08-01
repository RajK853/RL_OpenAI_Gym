import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import tensorflow_probability as tfp

from src.Layer import NeuralNetwork
from .base_policy import BasePolicy

MIN_LOG_STD = -20
MAX_LOG_STD = 2

# TODO: Use tf.keras.Model for policy network


class GaussianPolicy(BasePolicy):

    def __init__(self, *, layer_units, alpha=1e-1, lr=1e-3, **kwargs):
        super(GaussianPolicy, self).__init__(env=kwargs.pop("env"))
        self.alpha = alpha
        assert not self.discrete_action_space, "Action space for the Gaussian Policy must be continuous!"
        self.actions_ph = tf_v1.placeholder(tf.float32, shape=(None, *self.action_shape), name="action_ph")
        self.target_ph = tf_v1.placeholder(tf.float32, shape=(None,), name="target_ph")
        with tf_v1.variable_scope(self.scope):
            self.network = NeuralNetwork(name=self.scope, input_shape=self.obs_shape, layer_units=layer_units,
                                         output_size=2*self.action_size, **kwargs)
            # Split mean and std from the final layer
            mu = self.network.output[:, :self.action_size]
            mu = tf_v1.clip_by_value(mu, self.action_space.low, self.action_space.high)
            log_std = self.network.output[:, self.action_size:]
            log_std = tf_v1.clip_by_value(log_std, MIN_LOG_STD, MAX_LOG_STD)
            norm_dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf_v1.exp(log_std))
            # bijector = tfp.bijectors.Affine(shift=mu, scale_diag=tf_v1.exp(log_std))
        sample_action = norm_dist.sample()
        # sample_action = bijector.forward(norm_dist.sample())
        self.actions = tf.clip_by_value(sample_action, self.action_space.low, self.action_space.high)
        log_loss = norm_dist.log_prob(self.actions_ph)*self.target_ph
        entropy_loss = -self.alpha*norm_dist.entropy()
        self.loss = log_loss + entropy_loss
        # Optimizer Parameters
        self.optimizer = tf_v1.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.network.trainable_vars)
        self.scalar_summaries += ("mean_loss", )

    @property
    def mean_loss(self):
        return tf.reduce_mean(self.loss)

    def _action(self, sess, states, **kwargs):
        actions = sess.run(self.actions, feed_dict={self.network.inputs_ph: states})
        return actions

    def update(self, sess, states, actions, **kwargs):
        feed_dict = {self.network.inputs_ph: states, self.actions_ph: actions, self.target_ph: kwargs["targets"]}
        _,  loss, self.summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss
