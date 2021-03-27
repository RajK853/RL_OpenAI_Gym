import tensorflow.compat.v1 as tf_v1
import tensorflow_probability as tfp

from src.Network.neural_network import NeuralNetwork
from src.Network.utils import get_clipped_train_op
from .base_policy import BasePolicy
from src import Scheduler

tfd = tfp.distributions
tfb = tfp.bijectors

l2 = tf_v1.keras.regularizers.l2
DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "LayerNormalization"},
    {"type": "Dense", "units": 256, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "Dense", "units": 1}
]


class SquashBijector(tfp.bijectors.Bijector):
    """
    This squash bijector is derived from the given source:
    https://github.com/avisingh599/reward-learning-rl/blob/master/softlearning/distributions/squash_bijector.py
    """
    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(forward_min_event_ndims=0, validate_args=validate_args, name=name)

    def _forward(self, x):
        return tf_v1.tanh(x)

    def _inverse(self, y):
        return tf_v1.atanh(y)

    def _forward_log_det_jacobian(self, x):
        return 2.0*(tf_v1.log(2.0) - x - tf_v1.nn.softplus(-2.0*x))


class GaussianPolicy(BasePolicy):

    def __init__(self, *, lr_kwargs, layers=None, alpha=1e-3, learn_std=False, std_value=0.1, mu_range=None,
                 log_std_range=None, **kwargs):
        super(GaussianPolicy, self).__init__(**kwargs)
        scheduler = getattr(Scheduler, lr_kwargs.pop("type"))
        self.lr_scheduler = scheduler(**lr_kwargs)
        self.alpha = alpha
        self.mu_range = (-2.0, 2.0) if mu_range is None else mu_range
        self.log_std_range = (-20, 0.1) if log_std_range is None else log_std_range
        assert not self.discrete_action_space, "Action space for the Gaussian Policy must be continuous!"
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=(), name="lr_ph")
        self.targets_ph = tf_v1.placeholder("float32", shape=(None, 1), name="target_ph")
        # Create model
        if layers is None:
            layers = DEFAULT_LAYERS
        # layers[-1]["units"] = 2*self.action_size
        self.layers = layers
        self.base_model = NeuralNetwork(self.scope, input_shape=self.obs_shape, layers=self.layers)
        self.mu = tf_v1.keras.layers.Dense(self.action_size, activation=None, name="mu")(self.base_model.output)
        self.mu = tf_v1.clip_by_value(self.mu, *self.mu_range)
        if learn_std:
            self.log_std = tf_v1.keras.layers.Dense(self.action_size, name="log_std")(self.base_model.output)
            self.log_std = tf_v1.clip_by_value(self.log_std, *self.log_std_range)
            self.std = tf_v1.exp(self.log_std)
        else:
            self.std = tf_v1.constant([std_value]*self.action_size, dtype="float32")
        self.raw_action_model = tf_v1.keras.Model(inputs=[self.base_model.input], outputs=[self.mu, self.std])
        norm_dist = tfd.MultivariateNormalDiag(loc=tf_v1.zeros(self.action_size),
                                               scale_diag=tf_v1.ones(self.action_size))
        norm_dist = tfd.Independent(norm_dist)
        batch_size = tf_v1.shape(self.mu)[0]
        latents = norm_dist.sample(batch_size)
        bijector = tfb.Chain([SquashBijector(), tfb.Affine(shift=self.mu, scale_diag=self.std)])
        self.actions = bijector.forward(latents)
        self.deterministic_actions = tf_v1.tanh(self.mu)
        self.model = tf_v1.keras.Model(inputs=[self.base_model.input], outputs=[self.actions])
        # Loss parameters
        self._loss = None
        self.train_op = None
        # Summary parameters
        self.scalar_summaries += ("lr", )
        self.scalar_summaries_tf += ("loss", "mean_entropy", "min_mu", "mean_mu", "max_mu", "min_std", "mean_std",
                                     "max_std")
        self.histogram_summaries_tf += ("log_actions", "actions", "mu", "std")

    def log_prob(self, state, action, **kwargs):
        mu, std = self.raw_action_model(state)
        norm_dist = tfd.MultivariateNormalDiag(loc=tf_v1.zeros(self.action_size),
                                               scale_diag=tf_v1.ones(self.action_size))
        bijector = tfb.Chain([SquashBijector(), tfb.Affine(shift=mu, scale_diag=std)])
        dist = tfd.TransformedDistribution(distribution=norm_dist, bijector=bijector)
        log_pis = dist.log_prob(action)
        return log_pis

    @property
    def log_actions(self):
        return self.log_prob(self.base_model.input, self.actions)

    @property
    def lr(self):
        return self.lr_scheduler.value

    @property
    def mean_mu(self):
        return tf_v1.reduce_mean(self.mu)

    @property
    def min_mu(self):
        return tf_v1.reduce_min(self.mu)

    @property
    def max_mu(self):
        return tf_v1.reduce_max(self.mu)
    
    @property
    def mean_std(self):
        return tf_v1.reduce_mean(self.std)

    @property
    def min_std(self):
        return tf_v1.reduce_min(self.std)

    @property
    def max_std(self):
        return tf_v1.reduce_max(self.std)

    @property
    def mean_entropy(self):
        return tf_v1.reduce_mean(-self.log_actions)

    @property
    def loss(self):
        return tf_v1.reduce_mean(self._loss)

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    @property
    def trainable_vars(self):
        return self.model.trainable_variables

    def init_default_loss(self):
        log_loss = -self.log_actions * self.targets_ph
        entropy_loss = 0.0  # -self.alpha * self.dist.entropy()
        loss = log_loss + entropy_loss
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr_ph)
        train_op = get_clipped_train_op(loss, optimizer, var_list=self.trainable_vars, clip_norm=self.clip_norm)
        self.setup_loss(loss, train_op)

    def setup_loss(self, loss, train_op):
        self._loss = loss
        self.train_op = train_op

    def _action(self, sess, states, deterministic=False, **kwargs):
        action_op = self.deterministic_actions if deterministic else self.actions
        actions = sess.run(action_op, feed_dict={self.model.input: states})
        return actions

    def update(self, sess, feed_dict, **kwargs):
        train_ops = [self.train_op, self._loss]
        if self.summary_op is not None:
            train_ops.append(self.summary_op)
        results = sess.run(train_ops, feed_dict={self.lr_ph: self.lr, **feed_dict})
        if self.summary_op is not None:
            self.summary = results[-1]

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        self.lr_scheduler.increment()
