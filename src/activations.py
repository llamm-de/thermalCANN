import tensorflow as tf

# Collection of custom activation functions

def activation_exp_min_one(x):
    """
    Exponential activation function: exp(x) - 1 
    """
    return tf.math.exp(x) - 1.0

def activation_exp(x):
    """
    Exponential activation function: exp(x) 
    """
    return tf.math.exp(x)

def activation_squared(x):
    """
    Squared activation function 
    """
    return tf.math.pow(x,2.0)

def activation_ln_custom(x):
    """
    Custom logarithmic activation function 
    """
    return -1.0 * tf.math.log(1.0 - x)

def activation_ln(x):
    """
    Logarithmic activation function 
    """
    return tf.math.log(x)