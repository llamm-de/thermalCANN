import unittest

from src.data import load_treloar

class TestData(unittest.TestCase):
    def test_load_treloar(self):
        uni, ps, equi = load_treloar()
        self.assertEqual(uni[3][0], 1.24)
        self.assertEqual(uni[3][1], 0.23)
        self.assertEqual(uni[13][0], 5.36)
        self.assertEqual(uni[13][1], 1.94)

        self.assertEqual(ps[3][0], 1.21)
        self.assertEqual(ps[3][1], 0.24)
        self.assertEqual(ps[13][0], 4.96)
        self.assertEqual(ps[13][1], 1.79)

        self.assertEqual(equi[3][0], 1.12)
        self.assertEqual(equi[3][1], 0.24)
        self.assertEqual(equi[13][0], 3.75)
        self.assertEqual(equi[13][1], 1.72)
