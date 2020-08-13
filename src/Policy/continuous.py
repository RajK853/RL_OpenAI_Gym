import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from .base_policy import BasePolicy
from src.Layer import NeuralNetwork


class ContinuousPolicy(BasePolicy):

    def __init__(self, *, layer_units, lr=1e-3, **kwargs):
        self.scope = self.__class__.__name__
        super(ContinuousPolicy, self).__init__(env=kwargs.pop("env"))
        self.lr = lr
        self.layer_units = layer_units
        self.network_kwargs = kwargs
        self.network = self.build_network(self.scope)
        self._loss = None
        self.train_op = None
        # Summary parameters
        self.scalar_summaries += ("loss", )

    def build_network(self, scope):
        network = NeuralNetwork(scope, input_shape=self.obs_shape, layer_units=self.layer_units,
                                output_size=self.action_size, **self.network_kwargs)
        return network

    def set_loss(self, loss, optimizer=None):
        self._loss = loss
        optimizer = tf_v1.train.AdamOptimizer(learning_rate=self.lr) if optimizer is None else optimizer
        self.train_op = optimizer.minimize(self._loss, var_list=self.network.trainable_vars)

    @property
    def loss(self):
        return tf.reduce_mean(self._loss)

    def _action(self, sess, states, network=None, **kwargs):
        if network is None:
            network = self.network
        actions = sess.run(self.network.output, feed_dict={network.inputs_ph: states})
        return actions

    def update(self, sess, feed_dict=None, **kwargs):
        assert isinstance(feed_dict, dict), f"Invalid value received in feed_dict; {feed_dict}"
        _, loss, self.summary = sess.run([self.train_op, self._loss, self.summary_op], feed_dict=feed_dict)
        return loss
