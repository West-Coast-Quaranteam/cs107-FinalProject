import pytest
import numpy as np
# from ..apollo_ad import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apollo_ad import Variable


class TestScalar:

    def test_power(self):
        x = Variable(5)
        y = x ** 3
        assert y.var == 5 ** 3
        assert y.der == 5 * (3 ** 2)
