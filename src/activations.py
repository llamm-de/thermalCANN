import tensorflow as tf

# Collection of custom activation functions

def activation_exp(x, a = 1.0, b = 0.0):
    """
    Exponential activation function: a*exp(x)+b
    """
    return a*tf.math.exp(x) + b

def activation_squared(x, a = 1.0, b = 0.0):
    """
    Squared activation function 
    """
    return a*tf.math.pow(x,2.0) + b

def activation_ln(x, a = 1.0, b = 0.0, c = 0.0):
    """
    Logarithmic activation function 
    """
    return a*tf.math.log(x + c) + b