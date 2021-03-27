import numpy as np
import tensorflow.compat.v1 as tf_v1

from .base_policy import BasePolicy
from src.utils import get_scheduler
from src.Network.neural_network import NeuralNetwork
from src.Network.utils import get_clipped_train_op

l2 = tf_v1.keras.regularizers.l2
DEFAULT_LAYERS = [
    {"type": "Dense", "units": 126, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "LayerNormalization"},
    {"type": "Dense", "units": 126, "activation": "relu", "kernel_regularizer": l2(1e-8)},
    {"type": "Dense", "units": 1, "activation": "softmax", "name": "action_probs"},
]


class DiscretePolicy(BasePolicy):

    def __init__(self, *, lr_kwargs, layers=None, **kwargs):
        self.scope = self.__class__.__name__
        super(DiscretePolicy, self).__init__(**kwargs)
        self.lr_scheduler = get_scheduler(lr_kwargs)
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=[], name=f"{self.scope}/lr_ph")
        self.actions_ph = tf_v1.placeholder("int32", shape=(None, *self.action_shape), name=f"{self.scope}/actions_ph")
        self.targets_ph = tf_v1.placeholder("float32", shape=(None, 1), name=f"{self.scope}/targets_ph")
        # Create model
        if layers is None:
            layers = DEFAULT_LAYERS
        layers[-1]["units"] = self.action_size
        self.layers = layers
        self.model = NeuralNetwork(self.scope, input_shape=self.obs_shape, layers=self.layers)
        self.action_probs = self.model.output
        # Loss parameters
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

    @property
    def loss(self):
        return tf_v1.reduce_mean(self._loss)

    @property
    def trainable_vars(self):
        return self.model.trainable_vars

    def init_default_loss(self):
        hot_encoded = tf_v1.one_hot(self.actions_ph, self.action_size)
        log_actions = tf_v1.log(tf_v1.reduce_sum(hot_encoded*self.action_probs, axis=-1))
        loss = -log_actions * self.targets_ph
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr_ph)
        train_op = get_clipped_train_op(loss, optimizer=optimizer, var_list=self.model.trainable_vars,
                                        clip_norm=self.clip_norm)
        self.setup_loss(loss, train_op)

    def setup_loss(self, loss, train_op):
        self._loss = loss
        self.train_op = train_op

    def _action(self, sess, states, deterministic=False, **kwargs):
        probs = sess.run(self.action_probs, feed_dict={self.model.input: states})
        if deterministic:
            actions = np.array([np.argmax(p) for p in probs])
        else:
            actions = np.array([np.random.choice(self.action_size, p=p) for p in probs])
        return actions

    def update(self, sess, feed_dict, **kwargs):
        train_ops = [self.train_op, self.loss]
        if self.summary_op is not None:
            train_ops.append(self.summary_op)
        results = sess.run(train_ops, feed_dict={self.lr_ph: self.lr, **feed_dict})
        if self.summary_op is not None:
            self.summary = results[-1]

    def hook_after_epoch(self, **kwargs):
        super().hook_after_epoch(**kwargs)
        self.lr_scheduler.increment()
