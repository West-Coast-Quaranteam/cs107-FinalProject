import pytest
import numpy as np
from ..apollo_ad import *



class TestScalar:

    def test_pow(self):
        # test Variable raise to a constant
        x = Variable(5)
        y = x ** 3

        assert y.var == 5 ** 3
        assert y.der == 3 * (5 ** 2)

    def test_pow_imaginary(self):
        # test pow when base < 0 and exponent < 1
        with pytest.raises(ValueError):
            x = Variable(-1)
            y = x ** -0.5

    def test_rpow(self):
        # test constant raise to a variable
        x = Variable(3)
        y = 5 ** x
        assert y.var == 5 ** 3
        assert y.der == 125 * np.log(5)

    def test_rpow_imaginary(self):
        # test constant raise to a variable
        with pytest.raises(ValueError):
            x = Variable(0.5)
            y = (-2) ** x

    def test_sqrt(self):
        # test square root of a variable
        x = Variable(10.1)
        y = Variable.sqrt(x)
        assert y.var == np.sqrt(10.1)
        assert y.der == 1 / (2 * (np.sqrt(10.1)))

    def test_sqrt_constant(self):
        # test square root of a variable
        x = 12
        y = Variable.sqrt(x)
        assert y == np.sqrt(12)

    def test_sqrt_non_positive(self):
        # test square root of a non-positve variable
        with pytest.raises(ValueError):
            x = Variable(-10.1)
            y = Variable.sqrt(x)

    def test_exp(self):
        x = Variable(32)
        y = Variable.exp(x)
        assert y.var == np.exp(32)
        assert y.der == np.exp(32)

    def test_exp_constant(self):
        x = 32
        y = Variable.exp(x)
        assert y == np.exp(32)

    def test_log(self):
        x = Variable(14)
        y = Variable.log(x)
        assert y.var == np.log(14)
        assert y.der == 1 / 14

    def test_log_constant(self):
        x = 14
        y = Variable.log(x)
        assert y == np.log(14)

    def test_log_non_positive(self):
        with pytest.raises(ValueError):
            x = Variable(-14)
            y = Variable.log(x)

	def test_sin():
        # Expect value of 11.73, derivative of 3.0
        x = Variable(2)
        f = 3 * np.sin(x) + 3 * x + 3

        assert np.round(f.val, 2) == [11.73]
        assert np.round(f.der, 2) == [3.0]


		x2 = Variable(np.pi)
		f2 = 3 * np.sin(x2) +  3
		assert np.round(f2.val, 2) == [3.0]
		assert f2.der == [-3.0]


	def test_cos():
		# Expect value of -10pi, derivative of 0.0 (because of -sin(pi))
		x = Variable(np.pi)
		f = 5 * np.cos(x)+ np.cos(x) * 5
		assert f.val == [-10]
		assert abs(f.der) <= 1e-14

	def test_arcsin():
		x = Variable(0)
		f = np.arcsin(x)
		assert f.val == [0.0]
		assert f.der == [1.0]

		# Domain of arcsin(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = Variable(-1.01)
			np.arcsin(x)

	def test_arccos():
		x = Variable(0)
		f = np.arccos(x)
		assert f.val == [np.pi / 2]
		assert f.der == [-1.0]

		# Domain of arccos(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = Variable(1.01)
			np.arccos(x)