import tensorflow.compat.v1 as tf_v1
from src.utils import polyak_average
from .utils import create_feedforward_network


class NeuralNetwork:

    def __init__(self, scope, *, input_shape, layers):
        self.scope = scope
        self.input_shape = input_shape
        self.layers = layers
        self.input_tensor = tf_v1.keras.Input(shape=self.input_shape, dtype="float32", name=f"{self.scope}_input")
        self.output_tensor = create_feedforward_network(self.input_tensor, layers=self.layers)
        self.model = tf_v1.keras.Model(inputs=[self.input_tensor], outputs=[self.output_tensor], name=self.scope)
        # Weight update parameters
        self.weight_update_op = None
        self.tau_ph = tf_v1.placeholder("float32", name=f"{scope}/tau")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def input(self):
        return self.input_tensor

    @property
    def output(self):
        return self.output_tensor

    def init_weight_update_op(self, src_net):
        src_vars = src_net.trainable_vars
        target_vars = self.trainable_vars
        self.weight_update_op = tf_v1.group([tf_v1.assign(t_var, polyak_average(src_var, t_var, tau=self.tau_ph))
                                             for t_var, src_var in zip(target_vars, src_vars)])

    def update_weights(self, sess, tau):
        sess.run(self.weight_update_op, feed_dict={self.tau_ph: tau})

    def predict(self, inputs):
        if inputs.shape == self.input_shape:          # TODO: Ensure state dimension beforehand and remove this part
            inputs = inputs.reshape(1, -1)
        outputs = self.model.predict(inputs)
        return outputs

    @property
    def trainable_vars(self):
        return tuple(self.model.trainable_variables)
