import tensorflow as tf

# Collection of custom activation functions

def activation_exp(x):
    """
    Exponential activation function 
    """
    return tf.math.exp(x) - 1.0


def activation_ln(x):
    """
    Logarithmic activation function 
    """
    return -1.0 * tf.math.log(1.0 - x)