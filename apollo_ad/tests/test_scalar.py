import pytest
import numpy as np
from ..apollo_ad import Variable


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


    def test_sin(self):
        x = Variable(0)
        f = Variable.sin(x)

        assert f.var == 0.0
        assert f.der == [1.]


    def test_cos(self):
        x = Variable(0)
        f = Variable.cos(x)
        assert f.var == 1.0
        assert f.der == [-0.]

    def test_arcsin(self):
        x = Variable(0)
        f = Variable.arcsin(x)
        assert f.var == 0.0
        assert f.der == [1.]
        # -1<= x <=1

        with pytest.raises(ValueError):
            x = Variable(-2)
            f=Variable.arcsin(x)

    def test_arccos(self):
        x = Variable(0)
        f = Variable.arccos(x)
        assert f.var == 0.0
        assert f.der == [-1.]

        with pytest.raises(ValueError):
            x = Variable(2)
            f=Variable.arccos(x)