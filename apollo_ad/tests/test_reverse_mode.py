import pytest
import numpy as np
from ..reverse_mode import *


class TestReverse:

    def test_autoed_reverse_mode(self):
        x = Reverse_Mode(1)
        y = Reverse_Mode(2)
        inputs = [x, y]
        f = x*y+Reverse_Mode.exp(x*y)
        value, check = f.derivative(inputs)
        assert np.round(value, 4) == 9.3891
        assert np.round(check[0],4) == 16.7781 and np.round(check[1], 4) == 8.3891

        with pytest.raises(TypeError):
            x = Reverse_Mode('as')


    def test_mult_add_subtract_three_inputs(self):
        x = Reverse_Mode(1)
        y = Reverse_Mode(2)
        z = Reverse_Mode(3)
        f = 4 + x + 3 * y - 2 * z
        inputs = [x,y,z]
        value, check = f.derivative(inputs)
        assert np.round(value, 4) == 5.0000
        assert np.round(check[0], 4) == 1.0000
        assert np.round(check[1], 4) == 3.0000
        assert np.round(check[2], 4) == -2.0000


# HAVING ISSUES WITH NEGATION/SUBTRACTION
    def test_subtraction(self):
        x = Reverse_Mode(1)
        y = Reverse_Mode(2)
        z = Reverse_Mode(3)
        f = x - 3 * y + 2 * z - 10
        inputs = [x,y,z]
        value, check = f.derivative(inputs)
        assert np.round(value, 4) == -9.0000
        assert np.round(check[0], 4) == 1.0000
        assert np.round(check[1], 4) == -3.0000
        assert np.round(check[2], 4) == 2.0000

    def test_r_sub(self):
        x = Reverse_Mode(1)
        f = 2 - x
        inputs = [x]
        value, check = f.derivative(inputs)
        assert np.round(value, 4) == 1
        assert np.round(check[0], 4) == -1.0000

    def test_truediv_rtruediv(self):
        x = Reverse_Mode(3)
        f = x / 3

        inputs = [x]
        value, check = f.derivative(inputs)
        assert value == 1
        assert check[0] == 1/3

        x = Reverse_Mode(3)
        y = Reverse_Mode(4)
        f = x / y
        inputs = [x, y]
        value, check = f.derivative(inputs)
        assert value == 3 / 4
        assert check[0] == 1 / 4
        assert check[1] == -3 / 16

        x = Reverse_Mode(3)
        f = 2 / x
        inputs = [x]
        value, check = f.derivative(inputs)
        assert value == 2/3
        assert np.round(check[0], 4) == np.round(-2 / 9, 4)

    def test_eq(self):
        X = Reverse_Mode(3)
        Y = Reverse_Mode(3)
        flag = (X == Y)
        assert flag == True

        flag = (Reverse_Mode(3) == 3)
        assert flag == False

    def test_ne(self):
        X = Reverse_Mode(4)
        Y = Reverse_Mode(3)
        assert X != Y

        flag = (Reverse_Mode(3) != 3)
        assert flag == True

    def test_le_lt(self):
        X = Reverse_Mode(3)
        Y = Reverse_Mode(4)
        assert X < Y

        X = Reverse_Mode(3)
        Y = Reverse_Mode(4)
        assert X <= Y

        flag = (Reverse_Mode(3) < 3)
        assert flag == False

    def test_gt(self):
        X = Reverse_Mode(3)
        Y = Reverse_Mode(4)
        flag = (X > Y)
        assert flag == False

        flag = (Reverse_Mode(3) > 3)
        assert flag == False

    def test_ge(self):
        X = Reverse_Mode(3)
        Y = Reverse_Mode(4)
        flag = (X >= Y)
        assert flag == False

        flag = (Reverse_Mode(3) >= 3)
        assert flag == True

    def test_abs(self):
        x = Reverse_Mode(-3)
        inputs = [x]
        f = abs(x)
        value, grads = f.derivative(inputs)
        assert value == 3
        assert grads[0] == 1

    def test_pow(self):
        # test Variable raise to a constant
        x = Reverse_Mode(5)
        inputs = [x]
        f = x ** 3
        value, grads = f.derivative(inputs)
        assert value == 5 ** 3
        assert np.round(grads[0], 4) == np.round(3 * (5 ** 2), 4)

    def test_pow_imaginary_invalid_exponent(self):
        # test pow when base < 0 and exponent < 1
        with pytest.raises(ValueError):
            x = Reverse_Mode(-5)
            inputs = [x]
            f = x ** -0.5

        with pytest.raises(AttributeError):
            x = Reverse_Mode(5)
            y = x = Reverse_Mode(5)
            f = x ** y

    def test_rpow(self):
        # test constant raise to a variable
        x = Reverse_Mode(3)
        inputs = [x]
        f = 5 ** x
        value, grads = f.derivative(inputs)
        assert value == 5 ** 3
        assert np.round(grads, 4) == np.round(125 * np.log(5), 4)

    def test_rpow_imaginary(self):
        # test constant raise to a variable
        with pytest.raises(ValueError):
            x = Reverse_Mode(0.5)
            f = (-2) ** x

    def test_sqrt(self):
        # test square root of a variable
        x = Reverse_Mode(10.1)
        f = Reverse_Mode.sqrt(x)
        value, grads = f.derivative([x])
        assert np.round(value, 4) == np.round(np.sqrt(10.1), 4)
        assert np.round(grads[0], 4) == np.round(1 / (2 * (np.sqrt(10.1))), 4)

    def test_sqrt_constant(self):
        # test square root of a variable
        x = 12
        f = Reverse_Mode.sqrt(x)
        assert f == np.sqrt(12)

    def test_sqrt_non_positive(self):
        # test square root of a non-positve variable
        with pytest.raises(ValueError):
            x = Reverse_Mode(-10.1)
            f = Reverse_Mode.sqrt(x)

    def test_exp(self):
        x = Reverse_Mode(32)
        f = Reverse_Mode.exp(x)
        value, grads = f.derivative([x])
        assert value == np.exp(32)
        assert grads[0] == np.exp(32)

    def test_exp_constant(self):
        x = 32
        y = Reverse_Mode.exp(x)
        assert y == np.exp(32)

    def test_log(self):
        x = Reverse_Mode(14)
        f = Reverse_Mode.log(x)
        value, grads = f.derivative([x])
        assert value == np.log(14)
        assert grads[0] == 1 / 14

    def test_log_constant(self):
        x = 14
        f = Reverse_Mode.log(x)
        assert f == np.log(14)

    def test_log_non_positive(self):
        with pytest.raises(ValueError):
            x = Reverse_Mode(-14)
            f = Reverse_Mode.log(x)

    def test_sin(self):
        x = Reverse_Mode(0)
        f = Reverse_Mode.sin(x)
        v, g = f.derivative([x])
        assert v == 0.0
        assert g[0] == 1.

        # check constant
        assert Reverse_Mode.sin(2) == np.sin(2)

    def test_cos(self):
        x = Reverse_Mode(0)
        f = Reverse_Mode.cos(x)
        v, g = f.derivative([x])
        assert v == 1.0
        assert g[0] == 0

        # check constant
        assert Reverse_Mode.cos(2) == np.cos(2)

    def test_tangent_function(self):
        x = Reverse_Mode(np.pi)
        f = Reverse_Mode.tan(x)
        v, g = f.derivative([x])
        # However, if you specify a message with the assertion like this:
        # assert a % 2 == 0, "value was odd, should be even"
        # then no assertion introspection takes places at all and the message will be simply shown in the traceback.
        assert g == [1]

        x = Reverse_Mode(3 * np.pi / 2)
        with pytest.raises(ValueError):
            f = Reverse_Mode.tan(x)

        x = Reverse_Mode(2)
        f = 3 * Reverse_Mode.tan(x)
        v, g = f.derivative([x])
        assert v == 3 * np.tan(2) and np.round(g, 4) == [17.3232]

        x = Reverse_Mode(np.pi)
        f = Reverse_Mode.tan(x) * Reverse_Mode.tan(x)
        v, g = f.derivative([x])
        assert np.round(g[0], 5) == [0]

        # checking a constant
        assert Reverse_Mode.tan(3) == np.tan(3)

    def test_arcsin(self):
        x = Reverse_Mode(0)
        f = Reverse_Mode.arcsin(x)
        v, g = f.derivative([x])
        assert v == 0.0
        assert g == [1.]
        # -1<= x <=1

        with pytest.raises(ValueError):
            x = Reverse_Mode(-2)
            f = Reverse_Mode.arcsin(x)

        assert Reverse_Mode.arcsin(0.5) == np.arcsin(0.5)

    def test_arccos(self):
        x = Reverse_Mode(0)
        f = Reverse_Mode.arccos(x)
        v, g = f.derivative([x])
        assert v == 0.0
        assert g == [-1.]

        with pytest.raises(ValueError):
            x = Reverse_Mode(2)
            f = Reverse_Mode.arccos(x)

        assert Reverse_Mode.arccos(0.5) == np.arccos(0.5)

    def test_arctangent_function(self):
        x = Reverse_Mode(2)
        f = Reverse_Mode.arctan(x)
        v, g = f.derivative([x])
        assert v == np.arctan(2) and np.round(g[0],4) == 0.2

        # check a constant
        assert Reverse_Mode.arctan(3) == np.arctan(3)

    def test_sinh_function(self):
        x = Reverse_Mode(2)
        f = 2 * Reverse_Mode.sinh(x)
        v, g = f.derivative([x])
        assert np.round(g, 4) == [7.5244] and np.round(v, 4) == 7.2537

        # check a constant
        assert Reverse_Mode.sinh(3) == np.sinh(3)

    def test_cosh_function(self):
        x = Reverse_Mode(4)
        f = 3 * Reverse_Mode.cosh(x)
        v, g = f.derivative([x])
        assert np.round(v, 4) == 81.9247 and np.round(g, 4) == [81.8698]

        # check a constant
        assert Reverse_Mode.cosh(3) == np.cosh(3)

    def test_tanh(self):
        x = Reverse_Mode(3)
        f = 2 * Reverse_Mode.tanh(x)
        inputs = [x]
        value, grads = f.derivative(inputs)
        assert np.round(value, 4) == 1.9901
        assert np.round(grads[0], 4) == 0.0197

        # check constant
        assert Reverse_Mode.tanh(3) == np.tanh(3)