import tensorflow as tf
import tensorflow.compat.v1 as tf_v1

class Estimator():
    """
    Q-Value Estimator neural network.
    This network is used for both the local and target estimators.
    """
    def __init__(self, input_shape, output_size, scope="local_network"):
        """
        Constructor function
        args:
            input_shape (tuple) : Tuple of input shape
            output_size (int) : Number of outputs
            scope (str) : Variable scope name for the estimator
        """
        self.scope = scope
        self.input_shape = input_shape
        self.output_size = output_size

        # Writes Tensorboard summaries to disk
        with tf_v1.variable_scope(scope):
            # Build the graph
            self._build_model(input_shape, output_size)

    def _build_model(self, input_shape, output_size):
        """
        Builds the Tensorflow graph.
        args:
            input_shape (tuple) : Tuple of input shape
            output_size (int) : Number of outputs
        """
        # Placeholders for our input and output
        self.states_ph = tf_v1.placeholder(shape=[None, *input_shape], dtype=tf.float32, name="states")
        self.q_values_ph = tf_v1.placeholder(shape=[None, output_size], dtype=tf.float32, name="q_values")
        # Dense layers
        layer_out = tf.keras.layers.Dense(50, activation=tf.nn.relu)(self.states_ph)
        layer_out = tf.keras.layers.Dense(50, activation=tf.nn.relu)(layer_out)
        layer_out = tf.keras.layers.Dense(50, activation=tf.nn.relu)(layer_out)
        self.logits = tf.keras.layers.Dense(output_size)(layer_out)
        # Calculate the loss
        self.loss = tf_v1.losses.mean_squared_error(self.q_values_ph, self.logits)
        # Optimizer Parameters from original paper
        self.optimizer = tf_v1.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)

    def predict_one(self, sess, state):
        """
        Make prediction for one state
        args:
            sess (tf_v1.Session) : Tensorflow Session object
            state : State input of shape [1, state_shape]
        returns:
            Tensor : Tensor of shape [1, output_size] containing the estimated action values
        """
        return sess.run(self.logits, feed_dict={self.states_ph: state.reshape(1, *self.input_shape)})

    def predict_batch(self, sess, state):
        """
        Predicts action values of the batch.
        args:
          sess (tf_v1.Session) : Tensorflow session
          state : State input of shape [batch_size, state_shape]
        returns:
          Tensor : Tensor of shape [batch_size, output_size] containing the estimated action values.
        """
        return sess.run(self.logits, feed_dict={self.states_ph: state})

    def update(self, sess, state, pred_y):
        """
        Updates the estimator towards the given targets.
        args:
          sess (tf_v1.Session) : Tensorflow session object
          state : State input of shape [batch_size, state_shape]
          pred_y: Targets of shape [batch_size]
        returns:
          Tensor : Reduced average loss of the batch
        """
        feed_dict = {self.states_ph: state, self.q_values_ph: pred_y}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss