from tensorflow import keras
from .activations import activation_exp
import numpy as np


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
                                     activation=activation_exp,
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
    def __init__(self, size, l2_factor, name) -> None:
        super().__init__()
        self.size = size
        self.dense = keras.layers.Dense(size,
                                   kernel_initializer='glorot_normal',
                                   kernel_constraint=keras.constraints.NonNeg(),
                                   kernel_regularizer=keras.regularizers.l2(l2_factor),
                                   activation=None, 
                                   use_bias=False, 
                                   name=name)
        self.multi = keras.layers.Multiply()

    def call(self, tensors):
        tensor_nd = keras.layers.Lambda(lambda x: x*np.ones(self.size))(tensors[0])
        dense = self.dense(tensor_nd)
        return self.multi([dense, tensors[1]])


