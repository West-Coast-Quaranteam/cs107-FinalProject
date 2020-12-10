import pytest
import numpy as np
from ..apollo_ad import *


class TestScalar:

    def test_initializer(self):
        with pytest.raises(TypeError):
            x = Variable("hello")


    def test_add_radd(self):
        x = Variable(3, [1])
        y = x + 3

        assert y.var == 6
        assert y.der == np.array([1])
        
        x = Variable(3, [1])
        y = x + Variable(3, [1])

        assert y.var == 6
        assert y.der == np.array([2])

        x = Variable(3, [1, 0])
        y = x + Variable(3, [0, 1])

        assert y.var == 6
        assert np.array_equal(y.der, np.array([1, 1]))

        x = Variable(3, [1, 0])
        y = Variable(3, [0, 1]) + x

        assert y.var == 6
        assert np.array_equal(y.der, np.array([1, 1]))

    def test_mul_rmul(self):
        x = Variable(3, [1])
        y = x * 3

        assert y.var == 9
        assert y.der == np.array([3])

        x = Variable(3, [1])
        y = x * Variable(3, [1])

        assert y.var == 9
        assert y.der == np.array([6])

        x = Variable(3, [1, 0])
        y = x * Variable(3, [0, 1]) 

        assert y.var == 9
        assert np.array_equal(y.der, np.array([3, 3]))

        x = Variable(3, [1, 0])
        y = Variable(3, [0, 1]) * x

        assert y.var == 9
        assert np.array_equal(y.der, np.array([3, 3]))
    
    def test_sub_rsub(self):
        x = Variable(3, [1])
        y = x - 3

        assert y.var == 0
        assert y.der == np.array([1])

        x = Variable(3, [1])
        y = x - Variable(3, [1])

        assert y.var == 0
        assert y.der == np.array([0])

        x = Variable(4, [1, 0])
        y = x - Variable(3, [0, 1]) 

        assert y.var == 1
        assert np.array_equal(y.der, np.array([1, -1]))

        x = Variable(3, [1])
        y = 3 - x

        assert y.var == 0
        assert y.der == np.array([-1])

        x = Variable(3, [1])
        y = Variable(3, [1]) - x

        assert y.var == 0
        assert y.der == np.array([0])

        x = Variable(3, [1, 0])
        y = Variable(4, [0, 1]) - x

        assert y.var == 1
        assert np.array_equal(y.der, np.array([-1, 1]))

    def test_truediv_rtruediv(self):
        x = Variable(3, [1])
        y = x / 3

        assert y.var == 1
        assert y.der == np.array([1/3])

        x = Variable(3, [1])
        y = x / Variable(4, [1])

        assert y.var == 3/4
        assert y.der == np.array([1/16])

        x = Variable(3, [1, 0])
        y = x / Variable(4, [0, 1]) 

        assert y.var == 3/4
        assert np.array_equal(y.der, np.array([1/4, -3/16]))

        x = Variable(3, [1])
        y = 2 / x 

        assert y.var == 2/3
        assert y.der == np.array([-2/9])

        x = Variable(4, [1])
        y = Variable(3, [1]) / x

        assert y.var == 3/4
        assert y.der == np.array([1/16])

        x = Variable(4, [0, 1])
        y = Variable(3, [1, 0]) / x

        assert y.var == 3/4
        assert np.array_equal(y.der, np.array([1/4, -3/16]))

    def test_neg(self):
        x = Variable(3, [1])
        y = -x
        assert y.var == -3
        assert y.der == np.array([-1])

    def test_lt(self):
        X = Variable(3, 1)
        Y = Variable(4, [1])
        flag = (X < Y)
        assert flag == True

        flag = (Variable(3, [1]) < 3)
        assert flag == False
        
    def test_eq(self):
        X = Variable(3, 1)
        Y = Variable(3, [1])
        flag = (X == Y)
        assert flag == True

        flag = (Variable(3, [1]) == 3)
        assert flag == False

    def test_ne(self):
        X = Variable(3, 1)
        Y = Variable(3, [1])
        flag = (X != Y)
        assert flag == False

        flag = (Variable(3, [1]) != 3)
        assert flag == True

    def test_le(self):
        X = Variable(3, 1)
        Y = Variable(4, [1])
        flag = (X < Y)
        assert flag == True

        flag = (Variable(3, [1]) < 3)
        assert flag == False

    def test_gt(self):
        X = Variable(3, 1)
        Y = Variable(4, [1])
        flag = (X > Y)
        assert flag == False

        flag = (Variable(3, [1]) > 3)
        assert flag == False

    def test_ge(self):
        X = Variable(3, 1)
        Y = Variable(4, [1])
        flag = (X >= Y)
        assert flag == False

        flag = (Variable(3, [1]) >= 3)
        assert flag == True

    def test_abs(self):
        y = abs(Variable(-3, [-1]))
        assert y.var == 3
        assert y.der == np.array([1])

    def test_pow(self):
        # test Variable raise to a constant
        x = Variable(5)
        y = x ** 3

        assert y.var == 5 ** 3
        assert y.der == 3 * (5 ** 2)

    def test_pow_var(self):
        # test Variable raise to a Variable
        x = Variable(2)
        y = x ** (3 * x)

        assert y.var == 2 ** 6
        assert y.der == 192 + 192 * np.log(2)

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

        assert f.var == 3 * np.tan(2) and np.round(f.der,4) == [17.3232]

        x = Variable(np.pi)
        f = Variable.tan(x) * Variable.tan(x)

        assert np.round(f.der,5) == [0]

        # checking a constant
        assert Variable.tan(3) == np.tan(3)






    # can use these function below to run the code manually rather than with pytest
    def test_arctangent_function(self):
        x = Variable(2)
        f = Variable.arctan(x)

        assert f.der == [.2] and np.round(f.var,4) == 1.1071

        # check a constant
        assert Variable.arctan(3) == np.arctan(3)


    def test_sinh_function(self):
        x = Variable(2)
        f = 2 * Variable.sinh(x)

        assert np.round(f.der,4) == [7.5244] and np.round(f.var,4) == 7.2537

        # check a constant
        assert Variable.sinh(3) == np.sinh(3)


    def test_cosh_function(self):
        x = Variable(4)
        f = 3 * Variable.cosh(x)
        assert np.round(f.var,4) == 81.9247 and np.round(f.der,4) == [81.8698]

        # check a constant
        assert Variable.cosh(3) == np.cosh(3)


    def test_tanh_function(self):
        x = Variable(3)
        f = 2 * Variable.tanh(x)
        assert np.round(f.var,4) == 1.9901 and np.round(f.der,4) == [0.0197]

        # checking a constant
        assert Variable.tanh(3) == np.tanh(3)



    def test_sin(self):
        x = Variable(0)
        f = Variable.sin(x)

        assert f.var == 0.0
        assert f.der == [1.]

        # check constant
        assert Variable.sin(2) == np.sin(2)


    def test_cos(self):
        x = Variable(0)
        f = Variable.cos(x)
        assert f.var == 1.0
        assert f.der == [0.]

        # check constant
        assert Variable.cos(2) == np.cos(2)

    def test_arcsin(self):
        x = Variable(0)
        f = Variable.arcsin(x)
        assert f.var == 0.0
        assert f.der == [1.]
        # -1<= x <=1

        with pytest.raises(ValueError):
            x = Variable(-2)
            f=Variable.arcsin(x)

        assert Variable.arcsin(0.5) == np.arcsin(0.5)


    def test_arccos(self):
        x = Variable(0)
        f = Variable.arccos(x)
        assert f.var == 0.0
        assert f.der == [-1.]

        with pytest.raises(ValueError):
            x = Variable(2)
            f=Variable.arccos(x)

        assert Variable.arccos(0.5) == np.arccos(0.5)

    def test_functions(self):
        x = Variable(3, [1, 0, 0])
        y = Variable(4, [0, 1, 0])
        z = Variable(5, [0, 0, 1])
        f1 =  Variable.cos(x) + y ** 2 + Variable.sin(z)
        f2 = 2 * Variable.log(y) - Variable.sqrt(x)/3 
        z = Functions([f1, f2])

        assert np.array_equal(np.around(z.var, 4), np.array([14.0511, 2.1952]))
        assert np.array_equal(np.around(z.der, 4), np.array([[-0.1411, 8.0000, 0.2837], [-0.0962, 0.5000, 0.0000]]))

        with pytest.raises(TypeError):
            x = Variable(2)
            f=Variable.arccos(x)