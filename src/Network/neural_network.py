import tensorflow.compat.v1 as tf_v1
from src.utils import polyak_average
from .utils import create_feedforward_network


class NeuralNetwork:

    def __init__(self, scope, *, input_shapes, layers, preprocessors=None):
        self.scope = scope
        self.input_shapes = input_shapes
        self.layers = layers
        self.inputs = [tf_v1.keras.Input(shape=input_shape, dtype="float32", name=f"{self.scope}_input_{i}")
                       for i, input_shape in enumerate(self.input_shapes)]
        self.preprocessors = [None]*len(self.inputs) if preprocessors is None else preprocessors
        assert len(self.inputs) == len(self.preprocessors)
        self.model = self.init_model()
        # Weight update parameters
        self.weight_update_op = None
        self.tau_ph = tf_v1.placeholder("float32", name=f"{scope}/tau")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def init_model(self):
        inputs = [input_tensor if preprocessor is None else create_feedforward_network([input_tensor], layers=preprocessor)
                  for input_tensor, preprocessor in zip(self.inputs, self.preprocessors)]
        outputs = create_feedforward_network(inputs, layers=self.layers)
        return tf_v1.keras.Model(inputs=self.inputs, outputs=outputs, name=self.scope)

    @property
    def input(self):
        return self.model.input

    @property
    def output(self):
        return self.model.output

    def init_weight_update_op(self, src_net):
        src_vars = src_net.trainable_vars
        target_vars = self.trainable_vars
        self.weight_update_op = tf_v1.group([tf_v1.assign(t_var, polyak_average(src_var, t_var, tau=self.tau_ph))
                                             for t_var, src_var in zip(target_vars, src_vars)])

    def update_weights(self, sess, tau):
        sess.run(self.weight_update_op, feed_dict={self.tau_ph: tau})

    def predict(self, inputs):
        outputs = self.model.predict(inputs)
        return outputs

    @property
    def trainable_vars(self):
        return tuple(self.model.trainable_variables)


    def save(self, *args, **kwargs):
        self.model.save(*args, **kwargs)
