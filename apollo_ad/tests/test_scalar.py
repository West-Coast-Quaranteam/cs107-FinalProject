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


    def test_tangent_function(self):

        x = Variable(np.pi)
        f = Variable.tan(x)

        # However, if you specify a message with the assertion like this:
        # assert a % 2 == 0, "value was odd, should be even"
        # then no assertion introspection takes places at all and the message will be simply shown in the traceback.
        assert f.der == [1]

        x = Variable(3*np.pi/2)
        with pytest.raises(ValueError):
            f = Variable.tan(x)

        x = Variable(2)
        f = 3 * Variable.tan(x)

        assert f.var == -6.555119589784557 and f.der == [17.323197612125753]

        x = Variable(np.pi)
        f = Variable.tan(x) * Variable.tan(x)

        # I'm not sure if this should throw an error or not
        # when using AutoED this produces [-2048.537524357708], but the current value returned for us is [-2048.53752436]
        assert np.round(f.der,2) == [0]




    # can use these function below to run the code manually rather than with pytest
    def test_arctangent_function(self):
        x = Variable(2)
        f = Variable.arctan(x)

        assert f.der == [.2]


    def test_sinh_function(self):
        x = Variable(2)
        f = 2 * Variable.sinh(x)
        # on AutoED this gave 7.253720815694038 (only off by the final decimal, which was 7 vs. 8; this is where machine precision could have an issue)
        assert f.der == [7.524391382167263]


    def test_cosh_function(self):
        x = Variable(4)
        f = 3 * Variable.cosh(x)
        assert f.var == 81.92469850804946


    def test_tanh_function(self):
        x = Variable(3)
        f = 2 * Variable.tanh(x)
        assert f.var == 1.990109507373461 and f.der == [0.019732074330880384]
