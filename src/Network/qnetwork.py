import tensorflow.compat.v1 as tf_v1
from .neural_network import NeuralNetwork

regularizers = tf_v1.keras.regularizers
initializers = tf_v1.keras.initializers
constraints = tf_v1.keras.constraints

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
    {"type": "Dense", "units": 1},
]

class QNetwork(NeuralNetwork):
    """
    Q Network
    """
    def __init__(self, *, output_size, layers=None, **kwargs):
        """
        Constructor function
        args:
            input_shape (tuple) : Tuple of input shape
            output_size (int) : Number of outputs
            scope (str) : Variable scope name for the estimator
        """
        if layers is None:
            layers = DEFAULT_LAYERS
        layers[-1]["units"] = output_size    # Note: This works as long as the final layer is a Dense layer
        super(QNetwork, self).__init__(layers=layers, **kwargs)
        # Loss parameters
        self.loss = None
        self.train_op = None
        # Summary parameters
        self.summary_op = None
        self.summary = None
        self.scalar_summaries = ()
        self.histogram_summaries = ()
        self.scalar_summaries_tf = ("loss", )
        self.histogram_summaries_tf = ()

    def setup_loss(self, loss, train_op):
        self.loss = loss
        self.train_op = train_op

    def update(self, sess, feed_dict):
        train_ops = [self.train_op, self.loss]
        if self.summary_op is not None:
            train_ops.append(self.summary_op)
        results = sess.run(train_ops, feed_dict=feed_dict)
        if self.summary_op is not None:
            self.summary = results[-1]

    def init_summaries(self, tag="", force=False):
        if self.summary_op is None or force:
            _summaries = []
            for summary_type in ("scalar", "histogram"):
                summary_func = getattr(tf_v1.summary, summary_type)
                for summary_attr in getattr(self, f"{summary_type}_summaries_tf"):
                    attr = getattr(self, summary_attr)
                    _summaries.append(summary_func(f"{tag}/{self.scope}/{summary_attr}", attr))
            if _summaries:
                self.summary_op = tf_v1.summary.merge(_summaries)
