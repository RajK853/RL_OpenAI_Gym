import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

from src.Layer import NeuralNetwork
from src.Utils import get_space_size


class BasePolicy:

    def __init__(self, *, env):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.action_size = get_space_size(self.action_space)
        self.observation_size = get_space_size(self.observation_space)

    @property
    def action_shape(self):
        return self.action_space.shape

    @property
    def obs_shape(self):
        return self.observation_space.shape

    def action(self, sess, states, **kwargs):
        raise NotImplementedError

    def update(self, sess, states, actions, **kwargs):
        return 0.0

    def hook_before_action(self, **kwargs):
        pass

    def hook_after_action(self, **kwargs):
        pass

    def get_diagnostic(self):
        # TODO: Remove this later
        return {"eps": 0.0}


class UniformPolicy(BasePolicy):

    def __init__(self, *, layer_units, activation, scope_name="Uniform_Policy", **kwargs):
        with tf_v1.variable_scope(scope_name):
            super(UniformPolicy, self).__init__(**kwargs)
            self.actions_ph = tf_v1.placeholder(shape=[None, *self.action_shape], dtype=tf.int32, name="action_values")
            self.ndr_ph = tf_v1.placeholder(shape=[None, ], dtype=tf.float32, name="norm_discounted_rewards")
            self.policy_network = NeuralNetwork(scope_name, input_shape=self.obs_shape,
                                                layer_units=layer_units, activation=activation,
                                                output_size=self.action_size, output_activation=tf.nn.softmax)
            # Calculate the loss
            self.hot_encoded = tf_v1.one_hot(self.actions_ph, self.action_size)
            log_prob = tf_v1.keras.losses.categorical_crossentropy(self.hot_encoded, self.policy_network.output)
            self.loss = -tf.reduce_mean(log_prob * self.ndr_ph)
            # Optimizer Parameters
            var_list = self.policy_network.trainable_vars
            self.train_op = tf_v1.train.AdamOptimizer().minimize(self.loss, var_list=var_list)

    def action(self, sess, states, **kwargs):
        probs = self.policy_network.predict(sess, states) + 1e-10
        actions = [np.random.choice(self.action_size, p=p) for p in probs]
        return actions

    def update(self, sess, states, actions, **kwargs):
        feed_dict = {self.policy_network.input: states, self.actions_ph: actions,
                     self.ndr_ph: kwargs["norm_discount_reward"]}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
