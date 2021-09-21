import numpy as np
import tensorflow.compat.v1 as tf_v1

from .base_policy import BasePolicy
from src.utils import get_scheduler
from src.registry import registry
from src.Network.neural_network import NeuralNetwork
from src.Network.utils import get_clipped_train_op

regularizers = tf_v1.keras.regularizers

DEFAULT_KERNEL_KWARGS = {
    "kernel_regularizer": regularizers.l2(1e-3),
    "bias_regularizer": regularizers.l2(1e-6)
}

DEFAULT_LAYERS = [
    {"type": "Dense", "units": 256, **DEFAULT_KERNEL_KWARGS},
    {"type": "LayerNormalization"},
    {"type": "Activation", "activation": "relu"},
    {"type": "Dense", "units": 256, **DEFAULT_KERNEL_KWARGS},
    {"type": "LayerNormalization"},
    {"type": "Activation", "activation": "relu"},
    {"type": "Dense", "units": 1, "activation": "softmax"},
]

DEFAULT_KWARGS = {
    "lr_kwargs": {
        "type": "ConstantScheduler",
        "value": 0.0001,
    },
}


@registry.policy.register("discrete")
class DiscretePolicy(BasePolicy):
    PARAMETERS = BasePolicy.PARAMETERS.union({"lr_kwargs", "layers", "preprocessors"})

    def __init__(self, *, lr_kwargs=DEFAULT_KWARGS["lr_kwargs"], layers=None, preprocessors=None, **kwargs):
        super(DiscretePolicy, self).__init__(**kwargs)
        self.lr_kwargs = lr_kwargs
        self.lr_scheduler = get_scheduler(lr_kwargs)
        self.schedulers += (self.lr_scheduler, )
        # Placeholders
        self.lr_ph = tf_v1.placeholder("float32", shape=[], name=f"{self.scope}/lr_ph")
        if layers is None:
            layers = DEFAULT_LAYERS
        layers[-1]["units"] = self.action_size
        self.layers = layers
        self.preprocessors = preprocessors
        self.model = NeuralNetwork(self.scope, input_shapes=[self.obs_shape], layers=self.layers, preprocessors=self.preprocessors)
        # Loss parameters
        self._loss = None
        self.train_op = None
        # Summary parameters
        self.scalar_summaries_tf += ("loss", )
        self.scalar_summaries += ("lr", )

    def log_prob(self, state, action):
        action_probs = self.model(state)
        hot_encoded = tf_v1.one_hot(action, self.action_size)
        log_actions = tf_v1.log(tf_v1.reduce_sum(hot_encoded*action_probs, axis=-1))
        return log_actions

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

    def setup_loss(self, loss, train_op):
        self._loss = loss
        self.train_op = train_op

    def _action(self, sess, states, deterministic=False, **kwargs):
        # TODO: Check if model.predict is faster than sess.run(model.output,...)
        probs = self.model.predict(states)
        # probs = sess.run(self.model.output, feed_dict={self.model.input: states})
        if deterministic:
            actions = np.array([np.argmax(p) for p in probs])
        else:
            actions = np.array([np.random.choice(self.action_size, p=np.random.shuffle(p)) for p in probs])
        return actions

    def update(self, sess, feed_dict, **kwargs):
        train_ops = [self.train_op, self.loss]
        if self.summary_op is not None:
            train_ops.append(self.summary_op)
        results = sess.run(train_ops, feed_dict={self.lr_ph: self.lr, **feed_dict})
        if self.summary_op is not None:
            self.summary = results[-1]
