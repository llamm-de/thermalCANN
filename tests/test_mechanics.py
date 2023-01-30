import unittest

from src.mechanics import *
import tensorflow as tf

class TestMechanics(unittest.TestCase):

    def setUp(self):
        self.def_grad = tf.constant([[1.0, 0.0, 0.0],[0.5, 1.5, 0.0],[0.0, 0.0, 0.5]])

    def testInvariants(self):
        I1_expected = tf.constant(3.0)
        I2_expected = tf.constant(2.75)
        I1, I2 = Invariants()(self.def_grad)
        tf.debugging.assert_equal(I1, I1_expected)
        tf.debugging.assert_equal(I2, I2_expected)

    def testUniaxialDefGrad(self):
        stretch = tf.constant(2.0)
        def_grad = UniaxialDefGrad()(stretch)
        tf.debugging.assert_equal(def_grad, [[stretch, 0.0, 0.0],[0.0, 1/tf.sqrt(stretch), 0.0],[0.0, 0.0, 1/tf.sqrt(stretch)]])

    def testIsochoricVolumetricSplit(self):
        F_bar, J = IsochoricVolumetricSplit()(self.def_grad)
        tf.debugging.assert_equal(J, 0.75)
        tf.debugging.assert_near(F_bar, tf.constant([[1.1006424, 0.0, 0.0],[0.5503212, 1.650963, 0.0],[0.0, 0.0, 0.550321]]), )

    def testThermalSplit(self):
        del_theta = tf.constant(2.0)
        split = ThermalSplit()

        def_grad_mech = split(self.def_grad, del_theta)
        vartheta = tf.math.exp(split.w * del_theta)
        tf.debugging.assert_equal(def_grad_mech, self.def_grad/vartheta)

        del_theta = tf.constant(0.0)
        def_grad_mech = split(self.def_grad, del_theta)
        tf.debugging.assert_equal(def_grad_mech, self.def_grad)
        tf.debugging.assert_greater_equal(split.w, 0.0)

    def testRightCauchyGreen(self):
        right_cauchy_green = RightCauchyGreen()(self.def_grad)
        expected = tf.constant([[1.25, 0.75, 0.0],[0.75, 2.25, 0.0],[0.0, 0.0, 0.25]])
        tf.debugging.assert_equal(right_cauchy_green, expected)

    def testPushSecondPiolaKirchhoff(self):
        secondPK = tf.constant([[2.0, 0.0, 0.0],[0.25, 0.5, 0.0],[0.0, 0.0, 1.0]])
        tf.debugging.assert_equal(PushSecondPiolaKirchhoff()(secondPK, self.def_grad), tf.matmul(self.def_grad, secondPK))
