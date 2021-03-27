import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from src.utils import get_scheduler
from .base_policy import BasePolicy
from src.Network.neural_network import NeuralNetwork

l2 = tf_v1.keras.regularizers.l2
DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "LayerNormalization"},
    {"type": "Dense", "units": 256, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "Dense", "units": 1, "activation": "tanh"},
]


class ContinuousPolicy(BasePolicy):

    def __init__(self, *, lr_kwargs, layers=None, **kwargs):
        self.scope = self.__class__.__name__
        super(ContinuousPolicy, self).__init__(**kwargs)
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.lr_ph = tf_v1.placeholder("float32", shape=[], name=f"{self.scope}/lr_ph")
        if layers is None:
            layers = DEFAULT_LAYERS
        layers[-1]["units"] = self.action_size
        self.layers = layers
        self.model = NeuralNetwork(self.scope, input_shape=self.obs_shape, layers=self.layers)
        self._loss = None
        self.train_op = None
        # Summary parameters
        self.scalar_summaries_tf += ("loss", )
        self.scalar_summaries += ("lr", )

    @property
    def lr(self):
        return self.lr_scheduler.value

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    def setup_loss(self, loss, train_op):
        self._loss = loss
        self.train_op = train_op

    @property
    def trainable_vars(self):
        return self.model.trainable_vars

    @property
    def loss(self):
        return tf.reduce_mean(self._loss)

    def _action(self, sess, states, **kwargs):
        actions = sess.run(self.output, feed_dict={self.input: states})
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