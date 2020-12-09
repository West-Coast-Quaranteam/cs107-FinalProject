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

    def test_mult_add_three_inputs(self):
        x = Reverse_Mode(1)
        y = Reverse_Mode(2)
        z = Reverse_Mode(3)
        f = x + 3 * y + 2 * z
        inputs = [x,y,z]
        value, check = f.derivative(inputs)
        assert np.round(value, 4) == 13.0000
        assert np.round(check[0], 4) == 1.0000
        assert np.round(check[1], 4) == 3.0000
        assert np.round(check[2], 4) == 2.0000


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

    #     y = Reverse_Mode(2)
    #     f = x-4-y
    #     assert np.round(f.var,4) == -5.0000
    #     assert np.round(x.grad(),4) == 1.0000
    #     assert np.round(y.grad(), 4) == -1.0000

    def test_tanh(self):
        x = Reverse_Mode(3)
        f = 2 * x.tanh()
        inputs = [x]
        value, grads = f.derivative(inputs)
        assert np.round(value, 4) == 1.9901
        assert np.round(grads[0], 4) == 0.0197


if __name__ == "__main__":
    TestReverse.test_subtraction()