import unittest

from src.layers import Invariants, IsochoricVolumetricSplit, ThermalSplit, RightCauchyGreen, PushSecondPiolaKirchhoff
import tensorflow as tf

class TestLayers(unittest.TestCase):

    def setUp(self):
        self.tensor = tf.constant([[1.0,2.0],[3.0,4.0]])
        self.def_grad = tf.constant([[1.0, 0.0, 0.0],[0.5, 1.5, 0.0],[0.0, 0.0, 0.5]])

    def testInvariants(self):
        I1_expected = tf.constant(5.0)
        I2_expected = tf.constant(-2.0)
        I1, I2 = Invariants()(self.tensor)
        tf.debugging.assert_equal(I1, I1_expected)
        tf.debugging.assert_equal(I2, I2_expected)

    def testIsochoricVolumetricSplit(self):
        F_bar, J = IsochoricVolumetricSplit()(self.tensor)
        tf.debugging.assert_equal(J, 2.0)
        tf.debugging.assert_near(F_bar, tf.constant([[0.629961, 0.0],[0.0, 1.25992]]))

    def testThermalSplit(self):
        vartheta = tf.constant(2.0)
        def_grad_theta, def_grad_mech = ThermalSplit()(self.def_grad, vartheta)
        tf.debugging.assert_equal(def_grad_theta, vartheta * tf.eye(3,3))
        tf.debugging.assert_equal(def_grad_mech, self.def_grad/vartheta)

    def testRightCauchyGreen(self):
        right_cauchy_green = RightCauchyGreen()(self.def_grad)
        expected = tf.constant([[1.25, 0.75, 0.0],[0.75, 2.25, 0.0],[0.0, 0.0, 0.25]])
        tf.debugging.assert_equal(right_cauchy_green, expected)

    def testPushSecondPiolaKirchhoff(self):
        secondPK = tf.constant([[2.0, 0.0, 0.0],[0.25, 0.5, 0.0],[0.0, 0.0, 1.0]])
        tf.debugging.assert_equal(PushSecondPiolaKirchhoff()(secondPK, self.def_grad), tf.matmul(self.def_grad, secondPK))
