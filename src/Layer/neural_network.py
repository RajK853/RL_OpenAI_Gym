import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from tensorflow.keras.regularizers import l2


class NeuralNetwork:

    def __init__(self, name, *, input_shape, layer_units, output_size, input_type=tf.float32,
                 activation=tf.nn.relu, kernel_regularizer=None, output_activation=None, output_bias_regularizer=None,
                 output_kernel_regularizer=None):
        self.input_shape = input_shape
        self.output_size = output_size
        self.scope = name
        if kernel_regularizer is None:
            kernel_regularizer = l2(1e-3)
        self.inputs_ph = tf_v1.placeholder(input_type, shape=(None, *input_shape), name=f"{name}_inputs")
        self.network_output = self.build_network(layer_units, activation, kernel_regularizer, output_size,
                                                 output_activation, output_bias_regularizer, output_kernel_regularizer)
        self._trainable_vars = None

    def build_network(self, layer_units, activation, kernel_regulaizer, output_size, output_activation, output_bias_regularizer,
                      output_kernel_regularizer):
        output = self.inputs_ph
        with tf_v1.variable_scope(self.scope):
            for units in layer_units:
                output = tf.keras.layers.Dense(units, activation=activation,
                                               kernel_regularizer=kernel_regulaizer)(output)
            output = tf.keras.layers.Dense(output_size, activation=output_activation,
                                           kernel_regularizer=output_kernel_regularizer,
                                           bias_regularizer=output_bias_regularizer)(output)
        return output

    @property
    def input(self):
        return self.inputs_ph

    @property
    def output(self):
        return self.network_output

    def predict(self, sess, inputs):
        if inputs.shape == self.input_shape:
            inputs = inputs.reshape(1, *self.input_shape)
        outputs = sess.run(self.output, feed_dict={self.inputs_ph: inputs})
        return outputs

    @property
    def trainable_vars(self):
        if self._trainable_vars is None:
            self._trainable_vars = tf_v1.get_collection(tf_v1.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
            self._trainable_vars = sorted(self.trainable_vars, key=lambda var: var.name)
        return self._trainable_vars
