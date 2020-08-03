import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from .base_policy import BasePolicy
from src.Layer import NeuralNetwork


class DiscretePolicy(BasePolicy):

    def __init__(self, *, layer_units, lr=1e-3, **kwargs):
        self.scope = self.__class__.__name__
        super(DiscretePolicy, self).__init__(env=kwargs.pop("env"))
        self.actions_ph = tf_v1.placeholder(tf.int32, shape=(None, *self.action_shape), name="action_ph")
        self.target_ph = tf_v1.placeholder(tf.float32, shape=(None, ), name="target_ph")
        self.network = NeuralNetwork(self.scope, input_shape=self.obs_shape, layer_units=layer_units,
                                     output_size=self.action_size, **kwargs)
        self.logits = self.network.output
        self.action_probs = tf_v1.nn.softmax(self.logits, name="action_probs")
        # Calculate the loss
        self.hot_encoded = tf_v1.one_hot(self.actions_ph, self.action_size)
        self.log_prob = tf_v1.losses.softmax_cross_entropy(self.hot_encoded, self.logits,
                                                           reduction=tf_v1.losses.Reduction.NONE)
        self._loss = self.log_prob * self.target_ph
        # Optimizer parameters
        self.optimizer = tf_v1.train.AdamOptimizer(learning_rate=lr)
        self.train_op = self.optimizer.minimize(self._loss, var_list=self.network.trainable_vars)
        # Summary parameters
        self.scalar_summaries += ("loss", )
        self.histogram_summaries += ("log_prob", )

    @property
    def loss(self):
        return tf.reduce_mean(self._loss)

    def _action(self, sess, states, **kwargs):
        probs = sess.run(self.action_probs, feed_dict={self.network.inputs_ph: states})
        actions = np.array([np.random.choice(self.action_size, p=p) for p in probs])
        return actions

    def update(self, sess, states, actions, **kwargs):
        feed_dict = {self.network.input: states, self.actions_ph: actions, self.target_ph: kwargs["targets"]}
        _, loss, self.summary = sess.run([self.train_op, self._loss, self.summary_op], feed_dict=feed_dict)
        return loss
