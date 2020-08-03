import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.regularizers import l2
from src.utils import polyak_average


class NeuralNetwork:

    def __init__(self, scope, *, input_shape, layer_units, output_size, input_type=tf.float32,
                 layer_kwargs=None, output_kwargs=None):
        self.input_shape = input_shape
        self.output_size = output_size
        self.scope = scope
        self.inputs_ph = tf_v1.placeholder(input_type, shape=(None, *input_shape), name=f"{scope}_inputs")
        self.layers = []
        if layer_kwargs is None:
            layer_kwargs = {"activation": tf_v1.nn.relu, "kernel_regularizer": l2(1e-3)}
        if output_kwargs is None:
            output_kwargs = {}
        self.init_layers(layer_units, layer_kwargs, output_kwargs)
        self.network_output = self(self.inputs_ph)
        self._trainable_vars = None
        # Weight update parameters
        self.weight_update_op = None
        self.tau_ph = tf_v1.placeholder(tf.float32, name=f"{scope}/tau")

    def __call__(self, inputs):
        output = inputs
        with tf_v1.variable_scope(self.scope, reuse=tf_v1.AUTO_REUSE):
            for layer in self.layers:
                output = layer(output)
        return output

    def init_layers(self, layer_units, layer_kwargs, output_kwargs):
        self.layers.clear()
        with tf_v1.variable_scope(self.scope, reuse=tf_v1.AUTO_REUSE):
            for units in layer_units:
                layer = tf.keras.layers.Dense(units, **layer_kwargs)
                self.layers.append(layer)
            layer = tf.keras.layers.Dense(self.output_size, **output_kwargs)
            self.layers.append(layer)

    @property
    def input(self):
        return self.inputs_ph

    @property
    def output(self):
        return self.network_output

    def init_weight_update_op(self, src_net):
        if self.weight_update_op is None:
            target_vars = self.trainable_vars
            src_vars = src_net.trainable_vars
            self.weight_update_op = tf.group([tf_v1.assign(t_var, polyak_average(src_var, t_var, tau=self.tau_ph))
                                              for t_var, src_var in zip(target_vars, src_vars)])

    def update_weights(self, sess, tau):
        sess.run(self.weight_update_op, feed_dict={self.tau_ph: tau})

    def predict(self, sess, inputs):
        if inputs.shape == self.input_shape:
            inputs = inputs.reshape(1, *self.input_shape)
        outputs = sess.run(self.output, feed_dict={self.inputs_ph: inputs})
        return outputs

    @property
    def trainable_vars(self):
        if self._trainable_vars is None:
            self._trainable_vars = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        return self._trainable_vars
