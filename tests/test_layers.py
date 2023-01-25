import unittest

from src.layers import Invariants, IsochoricVolumetricSplit, ThermalSplit
import tensorflow as tf

class TestLayers(unittest.TestCase):

    def testInvariants(self):
        tensor = tf.constant([[1.0,2.0],[3.0,4.0]])
        I1_expected = tf.constant(5.0)
        I2_expected = tf.constant(-2.0)
        I1, I2 = Invariants()(tensor)
        tf.debugging.assert_equal(I1, I1_expected)
        tf.debugging.assert_equal(I2, I2_expected)

    def testIsochoricVolumetricSplit(self):
        tensor = tf.constant([[1.0,0.0],[0.0,2.0]])
        F_bar, J = IsochoricVolumetricSplit()(tensor)
        tf.debugging.assert_equal(J, 2.0)
        tf.debugging.assert_near(F_bar, tf.constant([[0.629961, 0.0],[0.0, 1.25992]]))

    def testThermalSplit(self):
        def_grad = tf.constant([[],[],[]])
        vartheta = tf.constant(2.0)
        def_grad_theta, def_grad_mech = ThermalSplit()(def_grad, vartheta)
        tf.debugging.assert_equal(def_grad_theta, vartheta * tf.eye(3,3))
        tf.debugging.assert_equal(def_grad_mech, def_grad/vartheta)