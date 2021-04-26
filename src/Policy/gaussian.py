import tensorflow.compat.v1 as tf_v1
import tensorflow_probability as tfp

from src.Network.neural_network import NeuralNetwork
from src.Network.utils import get_clipped_train_op
from src.utils import get_scheduler
from .base_policy import BasePolicy
from src import Scheduler

tfd = tfp.distributions
tfb = tfp.bijectors

regularizers = tf_v1.keras.regularizers
initializers = tf_v1.keras.initializers
constraints = tf_v1.keras.constraints

DEFAULT_KERNEL_KWARGS = {
    "kernel_regularizer": regularizers.l2(1e-3),
    "bias_regularizer": regularizers.l2(1e-6),
}

DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, **DEFAULT_KERNEL_KWARGS},
    {"type": "LayerNormalization"},
    {"type": "Activation", "activation": "relu"},
    {"type": "Dense", "units": 256, **DEFAULT_KERNEL_KWARGS},
    {"type": "LayerNormalization"},
    {"type": "Activation", "activation": "relu"},
]


class GaussianPolicy(BasePolicy):

    def __init__(self, *, lr_kwargs, layers=None, preprocessors=None, learn_std=True, std_value=0.1,
                 mu_range=None, log_std_range=None, **kwargs):
        super(GaussianPolicy, self).__init__(**kwargs)
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.schedulers += (self.lr_scheduler, )
        self.learn_std = learn_std
        self.mu_range = (-2.0, 2.0) if mu_range is None else mu_range
        self.log_std_range = (-10, 0.3) if log_std_range is None else log_std_range
        assert not self.discrete_action_space, "Action space for the Gaussian Policy must be continuous!"
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=(), name="lr_ph")
        # Create model
        if layers is None:
            layers = DEFAULT_LAYERS
        self.layers = layers
        self.preprocessors = preprocessors
        self.base_model = NeuralNetwork(self.scope, input_shapes=[self.obs_shape], layers=self.layers,
                                        preprocessors=self.preprocessors)
        self.mu = tf_v1.keras.layers.Dense(self.action_size, activation=None)(self.base_model.output)
        self.mu = tf_v1.clip_by_value(self.mu, *self.mu_range)
        if self.learn_std:
            self.log_std = tf_v1.keras.layers.Dense(self.action_size)(self.base_model.output)
            self.log_std = tf_v1.clip_by_value(self.log_std, *self.log_std_range)
            self.std = tf_v1.exp(self.log_std)
            self.raw_action_model = tf_v1.keras.Model(inputs=[self.base_model.input], outputs=[self.mu, self.std])
        else:
            self.std = tf_v1.constant([std_value]*self.action_size, dtype="float32")
            self.raw_action_model = tf_v1.keras.Model(inputs=[self.base_model.input], outputs=[self.mu])
        batch_size = tf_v1.shape(self.mu)[0]
        norm_dist = tfd.Normal(loc=tf_v1.zeros(self.action_size), scale=tf_v1.ones(self.action_size))
        z = norm_dist.sample(batch_size)
        raw_actions = self.mu + z*self.std          # Reparameterization trick
        self.actions = tf_v1.tanh(raw_actions)
        self.deterministic_actions = tf_v1.tanh(self.mu)
        self.model = tf_v1.keras.Model(inputs=[self.base_model.input], outputs=[self.actions])
        # Loss parameters
        self._loss = None
        self.train_op = None
        # Summary parameters
        self.scalar_summaries += ("lr", )
        self.scalar_summaries_tf += ("loss", "mean_log_actions", "min_mu", "mean_mu", "max_mu", "min_std", "mean_std",
                                     "max_std")
        self.histogram_summaries_tf += ("actions", "mu", "std", "log_actions")

    def mu_and_std(self, state):
        if self.learn_std:
            mu, std = self.raw_action_model(state)
        else:
            mu = self.raw_action_model(state)
            std = self.std
        return mu, std

    def log_prob(self, state, action):
        mu, std = self.mu_and_std(state)
        norm_dist = tfd.Normal(loc=mu, scale=std)
        log_probs = norm_dist.log_prob(tf_v1.atanh(action))
        log_probs -= tf_v1.log(1.0 - action**2 + 1e-8)
        log_probs = tf_v1.reduce_sum(log_probs, axis=-1, keepdims=True)
        return log_probs

    @property
    def log_actions(self):
        return self.log_prob([self.base_model.input], self.actions)

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
    def mean_log_actions(self):
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
