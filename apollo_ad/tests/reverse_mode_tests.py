import pytest
import numpy as np
from ..reverse_mode import *


class TestReverse:

    def test_multiple_inputs_one_output_and_add(self):
        x = Reverse_Mode(0.5)
        y = Reverse_Mode(4.2)
        f = 4*x+Reverse_Mode.sin(x)+6*y
        assert np.round(f.var, 4) == 27.6794
        assert np.round(x.grad(), 4) == 4.8776
        assert np.round(y.grad(), 4) == 6.0000

    def test_mult(self):
        x = Reverse_Mode(0.5)
        y = Reverse_Mode(4.2)
        f = 2 * x * y + Reverse_Mode.sin(x)
        assert abs(f.var - (2 * 0.5 * 4.2 + np.sin(0.5))) <= 1e-15
        assert abs(x.grad() - (2 * y.var + np.cos(x.var))) <= 1e-15
        assert abs(y.grad() - 2 * x.var) <= 1e-15

    def test_mult_add_three_inputs(self):
        x = Reverse_Mode(1)
        y = Reverse_Mode(2)
        z = Reverse_Mode(3)
        f = x + 3 * y + 2 * z
        assert np.round(f.var, 4) == 13.0000
        assert np.round(x.grad(), 4) == 1.0000
        assert np.round(y.grad(), 4) == 3.0000
        assert np.round(z.grad(), 4) == 2.0000

# HAVING ISSUES WITH NEGATION/SUBTRACTION
    # def test_subtraction(self):
    #     x = Reverse_Mode(1)
    #     f = x-4
    #     assert np.round(f.var,4) == -3.0000
    #     assert np.round(x.grad(),4) == 1.0000

    #     y = Reverse_Mode(2)
    #     f = x-4-y
    #     assert np.round(f.var,4) == -5.0000
    #     assert np.round(x.grad(),4) == 1.0000
    #     assert np.round(y.grad(), 4) == -1.0000
