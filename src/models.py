import tensorflow as tf
from tensorflow import keras

from layers import PsiIsoBlock
from mechanics import Invariants

# Set some parameters for network
L2 = 0.001


def originalCANN():
    """
    An implementation of the original CANN model from K. Linka & E. Kuhl (2023) 
    for uniaxial tension data.
    """

    stretch = keras.layers.Input(shape=(1,), name='Stretch')

    I1, I2 = Invariants()(stretch)

    psi = PsiIsoBlock(l2_factor = L2)(I1, I2)

    return keras.models.Model(inputs=[I1, I2], outputs=[psi], name = "Psi")
