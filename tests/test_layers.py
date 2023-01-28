import unittest
import tensorflow as tf
import numpy as np

from src.layers import MechanicsPreprocessBlock, PsiIsoBlock

class TestLayers(unittest.TestCase):

    def setUp(self):
        self.def_grad = tf.constant([[1.0, 0.0, 0.0],[0.5, 1.5, 0.0],[0.0, 0.0, 0.5]])

    def testMechanicsPreprocessingBlock(self):
        I1, I2, J = MechanicsPreprocessBlock()(self.def_grad)
        tf.debugging.assert_near(I1, 4.542802)
        tf.debugging.assert_near(I2, 4.586010)
        tf.debugging.assert_equal(J, 0.75)

    def testPsiIsoBlock(self):
        block = PsiIsoBlock()
        block.set_weights([np.ones(4), np.ones(4), np.ones(8)])
        
        # Test for no deformation
        I1 = tf.constant(3.0)
        I2 = tf.constant(3.0)
        psi = block(I1, I2)
        tf.debugging.assert_equal(psi, 0.0)

        # Test for deformation
        I1 = tf.constant(4.0)
        I2 = tf.constant(4.0)
        psi = block(I1, I2)
        tf.debugging.assert_equal(psi, 10.873127)