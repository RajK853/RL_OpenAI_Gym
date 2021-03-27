import copy
import tensorflow.compat.v1 as tf_v1


def get_clipped_train_op(loss, optimizer, var_list=None, clip_norm=None):
    gvs = optimizer.compute_gradients(loss, var_list=var_list)
    if clip_norm is not None:
        def clip(grad):
            if grad is None:
                return grad
            return tf_v1.clip_by_norm(grad, clip_norm)
        gvs = [(clip(grad), val) for grad, val in gvs]
    train_op = optimizer.apply_gradients(gvs)
    return train_op


def create_feedforward_network(input_tensor, layers):
    """
    Creates a feedforward network using the given input tensor
    :param input_tensor: (tensor) Input tensor to pass through the layers
    :param layers: (list) List of layer info as dicts in given format: [
                   {"type": "Conv2D", "filters": 32, "kernel_size": 3, "activation": "relu"},
                   ...,
                   {"type": "Dense", "units": 100, "activation": "relu"}]
    :returns: (tensor) Output tensor of the last layer
    """
    x = input_tensor
    for layer_dict in copy.deepcopy(layers):
        layer_type = layer_dict.pop("type")
        layer_class = getattr(tf_v1.keras.layers, layer_type)
        x = layer_class(**layer_dict)(x)
    return x
