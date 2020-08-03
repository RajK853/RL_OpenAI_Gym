import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
from .neural_network import NeuralNetwork


class QNetwork(NeuralNetwork):
    """
    Q Neural Network
    """
    DEFAULT_KWARGS = {"layer_units": (50, 50), "activation": tf.nn.relu}

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
        # Placeholders for targets
        self.targets_ph = tf_v1.placeholder(shape=[None, output_size], dtype=tf.float32, name=f"{scope}_targets")
        self.predictions = self.output
        self.loss = tf_v1.losses.mean_squared_error(self.targets_ph, self.predictions)
        self.train_op = tf_v1.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=self.trainable_vars)
        # Summary parameters
        self.summary_op = None
        self.summary = None
        self.scalar_summaries = ("loss", )
        self.histogram_summaries = ()

    def update(self, sess, states, targets):
        """
        Updates the estimator towards the given targets.
        args:
          sess (tf_v1.Session) : Tensorflow session object
          state : State input of shape [batch_size, state_shape]
          targets: Targets of shape [batch_size]
        returns:
          Tensor : Reduced average loss of the batch
        """
        feed_dict = {self.inputs_ph: states, self.targets_ph: targets}
        _, loss, self.summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss

    def init_summaries(self, tag="", force=False):
        if self.summary_op is None or force:
            _summaries = []
            for summary_type in ("scalar", "histogram"):
                summary_func = getattr(tf_v1.summary, summary_type)
                for summary_attr in getattr(self, f"{summary_type}_summaries"):
                    attr = getattr(self, summary_attr)
                    _summaries.append(summary_func(f"{tag}/{self.scope}/{summary_attr}", attr))
            if _summaries:
                self.summary_op = tf_v1.summary.merge(_summaries)
