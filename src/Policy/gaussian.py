import tensorflow.compat.v1 as tf_v1
import tensorflow_probability as tfp

from src.Layer import NeuralNetwork
from .base_policy import BasePolicy
import numpy as np

tfd = tfp.distributions
tfb = tfp.bijectors


class SquashBijector(tfp.bijectors.Bijector):
    """
    This squash bijector is derived from here:
    https://github.com/avisingh599/reward-learning-rl/blob/master/softlearning/distributions/squash_bijector.py
    """
    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)

    def _forward(self, x):
        return tf_v1.nn.tanh(x)

    def _inverse(self, y):
        return tf_v1.atanh(y)

    def _forward_log_det_jacobian(self, x):
        return 2. * (np.log(2.) - x - tf_v1.nn.softplus(-2. * x))


class GaussianPolicy(BasePolicy):

    def __init__(self, *, layer_units, alpha=1e-3, lr=1e-3, shift_scale=1.5,
                 min_log_scale=-20, max_log_scale=2, **kwargs):
        super(GaussianPolicy, self).__init__(env=kwargs.pop("env"))
        self.lr = lr
        self.alpha = alpha
        self.eps = np.finfo(np.float32).eps.item()
        self.shift_scale = shift_scale
        assert not self.discrete_action_space, "Action space for the Gaussian Policy must be continuous!"
        self.actions_ph = tf_v1.placeholder(tf_v1.float32, shape=(None, *self.action_shape), name="action_ph")
        self.targets_ph = tf_v1.placeholder(tf_v1.float32, shape=(None,), name="target_ph")
        with tf_v1.variable_scope(self.scope, reuse=tf_v1.AUTO_REUSE):
            self.network = NeuralNetwork(scope="neural_network", input_shape=self.obs_shape, layer_units=layer_units,
                                         output_size=2*self.action_size, **kwargs)
            self.shift = self.network.output[:, :self.action_size]
            # Limit shift in the range (-shift_scale, shift_scale) as large shift values produce NaNs in action
            self.shift = shift_scale*tf_v1.math.sin(self.shift)         
            self.log_scale = self.network.output[:, self.action_size:]
            self.log_scale = tf_v1.clip_by_value(self.log_scale, min_log_scale, max_log_scale)
            self.scale = tf_v1.exp(self.log_scale)
            batch_size = tf_v1.shape(self.shift)[0]
            norm_dist = tfd.MultivariateNormalDiag(loc=tf_v1.zeros(self.action_size),
                                                   scale_diag=tf_v1.ones(self.action_size))
            bijector = tfb.Chain([SquashBijector(),
                                  tfb.Affine(shift=self.shift, scale_diag=self.scale)])
            self.dist = tfd.TransformedDistribution(distribution=norm_dist, bijector=bijector)
            self.actions = self.dist.sample(batch_size)
            self.deterministic_actions = self.clip_action(self.shift, name="det_actions")
            self.log_actions = self.log_prob(self.actions_ph)
        # Loss parameters
        self._loss = None
        self.train_op = None
        # Summary parameters
        self.scalar_summaries += ("mean_loss", "mean_entropy", "min_shift", "mean_shift", "max_shift",
                                  "min_scale", "mean_scale", "max_scale")
        self.histogram_summaries += ("log_pi",)

    def clip_action(self, action, **kwargs):
        return tf_v1.tanh(action)

    def log_prob(self, action, **kwargs):
        return self.dist.log_prob(action, **kwargs)

    @property
    def log_pi(self):
        return self.log_prob(self.actions)

    @property
    def mean_shift(self):
        return tf_v1.reduce_mean(self.shift)

    @property
    def min_shift(self):
        return tf_v1.reduce_min(self.shift)

    @property
    def max_shift(self):
        return tf_v1.reduce_max(self.shift)
    
    @property
    def mean_scale(self):
        return tf_v1.reduce_mean(self.scale)

    @property
    def min_scale(self):
        return tf_v1.reduce_min(self.scale)

    @property
    def max_scale(self):
        return tf_v1.reduce_max(self.scale)

    @property
    def mean_entropy(self):
        return tf_v1.reduce_mean(-self.log_pi)

    @property
    def mean_loss(self):
        return tf_v1.reduce_mean(self.loss)

    @property
    def input(self):
        return self.network.input

    @property
    def output(self):
        return self.network.output

    def init_default_loss(self):
        log_loss = self.log_pi * self.targets_ph
        entropy_loss = -self.alpha * self.dist.entropy()
        loss = log_loss + entropy_loss
        self.set_loss(loss=loss)

    def set_loss(self, loss, train_op=None):
        self._loss = loss
        if train_op is None:
            optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = optimizer.minimize(self._loss, var_list=self.network.trainable_vars)
        else:
            self.train_op = train_op

    def _action(self, sess, states, deterministic=False, **kwargs):
        action_op = self.deterministic_actions if deterministic else self.actions
        actions = sess.run(action_op, feed_dict={self.network.inputs_ph: states})
        return actions

    def log_action(self, sess, states, actions, **kwargs):
        log_actions = sess.run(self.log_actions, feed_dict={self.network.inputs_ph: states, self.actions_ph: actions})
        return log_actions

    def update(self, sess, states, actions=None, feed_dict=None, **kwargs):
        _feed_dict = {self.network.inputs_ph: states}
        if actions is not None:
            _feed_dict.update({self.actions_ph: actions})
        if feed_dict is not None:
            _feed_dict.update(feed_dict)
        _, loss, self.summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=_feed_dict)
        return loss
