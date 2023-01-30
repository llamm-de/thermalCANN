import tensorflow as tf
from tensorflow import keras

from .layers import PsiIsoBlock
from .mechanics import Invariants, PushSecondPiolaKirchhoff

# Set some parameters for network
L2 = 0.001


def originalCANNpsi():
    """
    An implementation of the original CANN model from K. Linka & E. Kuhl (2023) 
    for uniaxial tension data.
    """

    def_grad = keras.layers.Input(shape=(3,3), name='Stretch')
    I1, I2 = Invariants()(def_grad)
    psi = PsiIsoBlock(l2_factor = L2)(I1, I2)

    return keras.models.Model(inputs=[def_grad], outputs=[psi], name = "Psi")
