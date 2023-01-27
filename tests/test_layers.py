import unittest
import tensorflow as tf

from src.layers import MechanicsPreprocessBlock

class TestLayers(unittest.TestCase):

    def setUp(self):
        self.def_grad = tf.constant([[1.0, 0.0, 0.0],[0.5, 1.5, 0.0],[0.0, 0.0, 0.5]])

    def testMechanicsPreprocessingBlock(self):
        I1, I2, J = MechanicsPreprocessBlock()(self.def_grad)
        tf.debugging.assert_near(I1, 4.542802)
        tf.debugging.assert_near(I2, 4.586010)
        tf.debugging.assert_equal(J, 0.75)