import pytest
import numpy as np
from ..reverse_mode import *


class TestReverse:

    def test_mult_and_add(self):
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