import tensorflow as tf
from tensorflow import keras
from .activations import activation_exp_min_one

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


class FunctionalLayer(keras.layers.Layer):
    """
    Custom functional layer for transformation of powers of invariants 
    using designated activation functions.
    """
    def __init__(self, identity_initializer, exponential_initializer, l2_factor, layer_id, node_id) -> None:
        super().__init__()
        self.identity = keras.layers.Dense(1,
                                  kernel_initializer=identity_initializer,
                                  kernel_constraint=keras.constraints.NonNeg(),
                                  kernel_regularizer=keras.regularizers.l2(l2_factor),
                                  use_bias=False, 
                                  activation=None,
                                  name=f"w{layer_id}{2*node_id-1}")
        self.exponential = keras.layers.Dense(1,
                                     kernel_initializer=exponential_initializer,
                                     kernel_constraint=keras.constraints.NonNeg(),
                                     kernel_regularizer=keras.regularizers.l2(l2_factor),
                                     use_bias=False, 
                                     activation=activation_exp_min_one,
                                     name=f"w{layer_id}{2*node_id}")

    def call(self, tensor):
        identity = self.identity(tensor)
        exponential = self.exponential(tensor)
        return keras.layers.concatenate([identity, exponential], axis=1)


class MultiplyLayer(keras.layers.Layer):
    """
    Custom multiplication layer class.
    Multiplies a scalar input from the network with a tensor.
    """
    def __init__(self, l2_factor, name) -> None:
        super().__init__()
        self.dense = keras.layers.Dense(1,
                                   kernel_initializer='glorot_normal',
                                   kernel_constraint=keras.constraints.NonNeg(),
                                   kernel_regularizer=keras.regularizers.l2(l2_factor),
                                   activation=None, 
                                   use_bias=False, 
                                   name=name)
        self.multi = keras.layers.Multiply()

    def call(self, tensors):
        theta_weighted = self.dense(tensors[0])
        return keras.layers.Lambda(lambda x: x[0]*x[1])([theta_weighted, tensors[1]])


