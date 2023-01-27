import tensorflow as tf
from tensorflow import keras

from .mechanics import IsochoricVolumetricSplit, Invariants, RightCauchyGreen
from .activations import activation_exp


class MechanicsPreprocessBlock(keras.layers.Layer):
    """
    Block for preprocessing of mechanics. First apply isochoric-volumetric
    split. Next, calculate the isochoric right Cauchy Green tensor and its invariants.

    Inputs:
        def_grad - Deformation gradient (mechanical part)

    Outputs:
        I1 - First invariant of isochoric right Cauchy Green tensor 
        I2 - Second invariant of isochoric right Cauchy Green tensor
        J -  Determinant of isochoric deformation gradient
    """
    def __init__(self) -> None:
        super().__init__()

    def call(self, def_grad):
        def_grad_iso, J = IsochoricVolumetricSplit()(def_grad)
        right_cauchy_green_iso = RightCauchyGreen()(def_grad_iso)
        I1, I2 = Invariants()(right_cauchy_green_iso)
        return I1, I2, J


class PsiIsoBlock(keras.layers.Layer):
    """
    Blog for computing isotropic strain energy density from invariants of 
    isochoric right Cauchy Green tensor.

    Inputs:
        I1 - First invariant of isochoric right Cauchy Green tensor
        I2 - Second invariant of isochoric right Cauchy Green tensor

    Outputs:
        PsiIso - Isochoric strain energy function
    """
    def __init__(self, max_power=2, l2_factor = 0.001) -> None:
        super().__init__()
        self.powers = tf.range(1,max_power+1,dtype='float32')
        self.w_identity = self.add_weight(shape = (2*max_power,), 
                                 initializer = keras.initializers.GlorotNormal(), 
                                 constraint = tf.keras.constraints.NonNeg(), 
                                 regularizer = keras.regularizers.l2(l2_factor),
                                 trainable = True)                         
        self.w_exp = self.add_weight(shape = (2*max_power,), 
                                 initializer = keras.initializers.RandomUniform(minval=0.0, maxval=0.00001), 
                                 constraint = tf.keras.constraints.NonNeg(), 
                                 regularizer = keras.regularizers.l2(l2_factor),
                                 trainable = True)
        self.w_psi = self.add_weight(shape = (2*2*max_power,), 
                                 initializer = keras.initializers.GlorotNormal(), 
                                 constraint = tf.keras.constraints.NonNeg(), 
                                 regularizer = keras.regularizers.l2(l2_factor),
                                 trainable = True)  
        self.activation_exp = keras.layers.Lambda(lambda x: activation_exp(x, b=-1.0))

    def call(self, I1, I2):
        
        # Compute invariants in reference
        I1_ref = keras.layers.Lambda(lambda x: x-3)(I1)
        I2_ref = keras.layers.Lambda(lambda x: x-3)(I2)

        # Raise reference invariants to powers      
        I1_powers = tf.math.pow(I1_ref, self.powers)
        I2_powers = tf.math.pow(I2_ref, self.powers)
        powers = tf.concat(I1_powers, I2_powers,0)
        
        # Multiply by weights
        powers_identity = keras.layers.Lambda(lambda x: x[0]*x[1])(powers, self.w_identity)
        powers_exp = keras.layers.Lambda(lambda x: x[0]*x[1])(powers, self.w_exp)
        
        # Apply activation functions to powers of invariants
        powers_exp = self.activation_exp(powers_exp)

        # Concat results
        active_results = tf.concat(powers_identity, powers_exp, 0)

        # Multiply results with weights and add to strain energy tf.tensordot
        return tf.tensordot([active_results, self.w_psi], 0)
        
        

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