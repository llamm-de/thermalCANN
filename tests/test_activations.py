import unittest

from src.activations import activation_exp, activation_exp_min_one, activation_ln
import tensorflow as tf

class TestActivations(unittest.TestCase):

    def test_activation_exp_min_one(self):
        self.assertEqual(activation_exp_min_one(0.0), 0.0)
        self.assertEqual(activation_exp_min_one(1.0), tf.math.exp(1.0) - 1.0)

    def test_activation_exp(self):
        self.assertEqual(activation_exp(0.0), 1.0)
        self.assertEqual(activation_exp(1.0), tf.math.exp(1.0))

    def test_activation_ln(self):
        self.assertEqual(activation_ln(0.0), 0.0)
        self.assertEqual(activation_ln(1.0), -1.0 * tf.math.log(0.0))


