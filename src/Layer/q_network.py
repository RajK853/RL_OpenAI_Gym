import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from .neural_network import NeuralNetwork


class QNetwork(NeuralNetwork):
    """
    Q Neural Network
    """
    DEFAULT_KWARGS = {"layer_units": (50, 50, 50), "activation": tf.nn.relu}

    def __init__(self, input_shape, output_size, scope="local_network", lr=0.001, **kwargs):
        """
        Constructor function
        args:
            input_shape (tuple) : Tuple of input shape
            output_size (int) : Number of outputs
            scope (str) : Variable scope name for the estimator
        """
        kwargs = kwargs or self.DEFAULT_KWARGS
        super(QNetwork, self).__init__(scope, input_shape=input_shape, output_size=output_size, **kwargs)
        # Placeholders for targets and actions
        self.targets_ph = tf_v1.placeholder(shape=[None, ], dtype=tf.float32, name=f"{scope}_targets")
        self.actions_ph = tf_v1.placeholder(shape=[None, ], dtype=tf.int32, name=f"{scope}_actions")
        # Get the predictions for the chosen actions only
        batch_size = tf.shape(self.targets_ph)[0]
        gather_indices = tf.range(batch_size) * tf.shape(self.output)[1] + self.actions_ph
        self.action_predictions = tf.gather(tf.reshape(self.output, [-1]), gather_indices)
        self.loss = tf_v1.losses.mean_squared_error(self.targets_ph, self.action_predictions)
        self.train_op = tf_v1.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=self.trainable_vars)

    def update(self, sess, states, targets, actions):
        """
        Updates the estimator towards the given targets.
        args:
          sess (tf_v1.Session) : Tensorflow session object
          state : State input of shape [batch_size, state_shape]
          targets: Targets of shape [batch_size]
        returns:
          Tensor : Reduced average loss of the batch
        """
        feed_dict = {self.inputs_ph: states, self.targets_ph: targets, self.actions_ph: actions}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss
