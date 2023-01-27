import tensorflow as tf
from tensorflow import keras


class CustomDense(keras.layers.Layer):
    """
    Convenience layer for dense layers
    """
    def __init__(self, activation, initializer='glorot_normal', l2_factor=0.001, name='customDense') -> None:
        super().__init__()
        self.dense = keras.layers.Dense(1,
                                        kernel_initializer=initializer,
                                        kernel_constraint=keras.constraints.NonNeg(),
                                        kernel_regularizer=keras.regularizers.l2(l2_factor),
                                        use_bias=False, 
                                        activation=activation,
                                        name=name)

    def call(self, tensor):
        return self.dense(tensor)


# class MechanicsPreprocessBlock(keras.layers.Layer):


class FunctionalBlock(keras.layers.Layer):
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


class MultiplyBlock(keras.layers.Layer):
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


