import tensorflow as tf
from tensorflow import keras

class Invariants(keras.layers.Layer):
    """
    Custom layer to calculate the invariants of a tensor
    """
    def __init__(self) -> None:
        super().__init__()

    def call(self, tensor):
        two = tf.constant(2.0)
        invariant_1 = tf.linalg.trace(tensor)
        tensor_squared = tf.linalg.matmul(tensor, tensor)
        trace_squared = tf.math.pow(tf.linalg.trace(tensor), two)
        trace_tensor_squared = tf.linalg.trace(tensor_squared)
        invariant_2 = (trace_squared - trace_tensor_squared)/2
        return invariant_1, invariant_2

class IsochoricVolumetricSplit(keras.layers.Layer):
    """
    Custom layer to calculate isochoric-volumetric split of deformation gradient
    """
    def __init__(self) -> None:
        super().__init__()

    def call(self, def_grad):
        J = tf.linalg.det(def_grad)
        def_grad_iso = tf.math.pow(J, -2.0/3.0) * def_grad
        return def_grad_iso, J

class ThermalSplit(keras.layers.Layer):
    """
    Custom layer to calculate the thermal split of the deformation gradient
    """
    def __init__(self) -> None:
        super().__init__()
    
    def call(self, def_grad, vartheta):
        def_grad_theta = vartheta * tf.eye(3,3)
        def_grad_mech = def_grad / vartheta
        return def_grad_theta, def_grad_mech

class RightCauchyGreen(keras.layers.Layer):
    """
    Custom layer to calculate the right Cauchy Green tensor from deformation gradient
    """
    def __init__(self) -> None:
        super().__init__()

    def call(self, def_grad):
        return tf.matmul(tf.transpose(def_grad), def_grad)

class PushSecondPiolaKirchhoff(keras.layers.Layer):
    """
    Custom layer to calculate push forward of second PK to first PK
    """
    def __init__(self) -> None:
        super().__init__()

    def call(self, second_pk, def_grad):
        return tf.matmul(def_grad, second_pk)