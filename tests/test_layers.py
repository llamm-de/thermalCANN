import unittest

from src.layers import InvariantLayer
import tensorflow as tf

class TestLayers(unittest.TestCase):

    def testInvariantLayer(self):
        tensor = tf.constant([[1.0,2.0],[3.0,4.0]])
        I1_expected = tf.constant(5.0)
        I2_expected = tf.constant(-2.0)
        I1, I2 = InvariantLayer()(tensor)
        tf.debugging.assert_equal(I1, I1_expected)
        tf.debugging.assert_equal(I2, I2_expected)