import tensorflow as tf
from tensorflow import keras
from layers import FunctionalLayer, MultiplyLayer
import numpy as np

# Set some parameters for network
IDENTITY_INITIALIZER = 'glorot_normal'
EXPONENTIAL_INITIALIZER = keras.initializers.RandomUniform(minval=0.0, maxval=0.00001)
L2 = 0.001


def thermal_strain_energy_CANN():
    """
    Extension of the standard CANN from Linka & Kuhl to also consider temperature.
    """

    # Input layer
    invariant_1 = keras.Input(shape=(1,), name='Invariant_1')
    invariant_2 = keras.Input(shape=(1,), name='Invariant_2')
    theta = keras.Input(shape=(1,), name='Theta')

    # Transformation layer
    invariant_1_ref = keras.layers.Lambda(lambda x: (x-3))(invariant_1)
    invariant_2_ref = keras.layers.Lambda(lambda x: (x-3))(invariant_2)

    # Exponentiation layer (Only using to the power of 2 for now)
    invariant_1_ref_sq = keras.layers.Lambda(lambda x: tf.math.square(x))(invariant_1_ref)
    invariant_2_ref_sq = keras.layers.Lambda(lambda x: tf.math.square(x))(invariant_2_ref)

    invariant_powers = [invariant_1_ref, invariant_1_ref_sq, invariant_2_ref, invariant_2_ref_sq]

    # Functional layer
    invariant_activations = []
    for id, invariant_power in enumerate(invariant_powers):
        x = FunctionalLayer(IDENTITY_INITIALIZER,EXPONENTIAL_INITIALIZER, L2, 1, id+1)(invariant_power)
        invariant_activations.append(x)

    functional_layer = keras.layers.concatenate(invariant_activations, axis=1)

    # Thermal multiplication layer
    size = len(invariant_powers) * 2
    theta_multiply = MultiplyLayer(size, L2, 'w2x')([theta, functional_layer])

    # Strain energy layer
    strain_energy = keras.layers.Dense(1,
                                       kernel_initializer='glorot_normal',
                                       kernel_constraint=keras.constraints.NonNeg(),
                                       kernel_regularizer=keras.regularizers.l2(L2),
                                       use_bias=False, 
                                       activation=None,
                                       name='w3x')(theta_multiply)

    model = keras.models.Model(inputs=[invariant_1, invariant_2, theta], outputs=[strain_energy], name='Strain_Energy')

    return model
