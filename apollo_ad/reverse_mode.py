import numpy as np
import cProfile


class Reverse_Mode:
    def __init__(self, var):
        self.var = var
        # need this because we need to store a form of the computational graph
        self.child = []
        self.der = None

    def gradient(self):
        # when calling this on the first variable, it should be that everything is self.der = None except for the output function, which has been initialized to 1
        # if self.der is not None, then the derivative has already been calculated for that variable, and we should just return it rather than calculating everything again

        if self.der is None:
            # I'm using this based on the equation for lecture 12 we had in class
            df_dui = 0
            for duj_dui, df_duj in self.child:
                # I need this += so that I can handle when the node has more than one child
                df_dui += duj_dui * df_duj.gradient()
            self.der = df_dui
        return self.der

    def derivative(self, inputs, seed = 1):
        self.der = seed
        value = self.var
        derivatives = np.array([i.gradient() for i in inputs])
        self._reset(inputs)

        return value, derivatives

    @staticmethod
    def _reset(inputs):
        for i in inputs:
            i.der = None

    def __add__(self, other):
        try:
            f = Reverse_Mode(self.var + other.var)
            other.child.append((1.0, f))
            self.child.append((1.0, f))
        except AttributeError:
            f = Reverse_Mode(self.var + other)
            self.child.append((1.0, f))
        return f

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        """Dunder method for taking the negative
         INPUTS
         =======
         self: Reverse_Mode object

         RETURNS
         ========
         output: Reverse_Mode, a new Reverse_Mode in negation

         EXAMPLES
         =========
         >>>
         >>>

        """

        f = Reverse_Mode(-self.var)
        self.child.append((-1, f))
        return f

    def __mul__(self, other):
        try:
            f = Reverse_Mode(self.var * other.var)
            self.child.append((other.var, f))
            other.child.append((self.var, f))
        except AttributeError:
            f = Reverse_Mode(self.var * other)
            self.child.append((other, f))
        return f

    def __rmul__(self, other):
        return self.__mul__(other)

# HAVING ISSUES WITH NEGATION/SUBTRACTION
# UPDATE: ISSUES FIXED. BY HAOXIN

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __truediv__(self, other):
        """Dunder method for dividing another variable or scalar/vector
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to divide self.var.

         RETURNS
         ========
         output: Variable, a new variable that self.var divides `other`.

         EXAMPLES
         =========
         divides scalar
         >>> x = Variable(3, [1])
         >>> x / 3
         Variable(1, 1/3)

         divides another variable - X
         >>> x = Variable(3, [1])
         >>> x / Variable(4, [1])
         Variable(3/4, [1/16])

         divides another variable - Y
         >>> x = Variable(3, [1, 0])
         >>> x / Variable(4, [0, 1])
         Variable(3/4, [1/4, -3/16])
        """
        try:
            other_new = Reverse_Mode(1 / other.var)
            other.child.append((-other.var ** -2, other_new))
            return self * other_new

        except AttributeError:
            f = Reverse_Mode(self.var / other)
            der = 1 / other
            self.child.append((der, f))

        return f

    def __rtruediv__(self, other):
        """Dunder method for being divided by another variable or scalar/vector
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to be divided by self.var.

         RETURNS
         ========
         output: Variable, a new variable 'other' divides self.var.

         EXAMPLES
         =========
         divided by scalar
         >>> x = Variable(3, [1])
         >>> 2 / x
         Variable(2/3, [-2/9])

         divided by another variable - X
         >>> x = Variable(4, [1])
         >>> Variable(3, [1]) / x
         Variable(3/4, [1/16])

         divided by another variable - Y
         >>> x = Variable(4, [0, 1])
         >>> Variable(3, [1, 0]) / x
         Variable(3/4, [1/4, -3/16])
        """
        f = Reverse_Mode(other / self.var)
        der = other * (-self.var ** (-2)) * 1
        self.child.append((der, f))

        return f

    def __eq__(self, other):
        """Dunder method for checking equality
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to be checked if it is equal

         RETURNS
         ========
         output: bool, True/False, Equal/Not Equal.

         EXAMPLES
         =========
         >> Variable(3, [1]) == 3
         False

         >>> X = Variable(3, 1)
         >>> Y = Variable(3, [1])
         >>> X == Y
         True
        """
        try:
            out = (self.var == other.var)
        except AttributeError:
            print('A scalar and a Variable type does not equal')
            out = False
        return out

    def __ne__(self, other):
        """Dunder method for checking inequality
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to be checked if it is not equal

         RETURNS
         ========
         output: bool, True/False, Not Equal/ Equal.

         EXAMPLES
         =========
         >> 3 != Variable(3, [1])
         True

         >>> X = Variable(3, 1)
         >>> Y = Variable(3, [1])
         >>> Y != X
         False
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        """Dunder method for less than
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable

         RETURNS
         ========
         out: bool, True/False, less than/ not less than.

         EXAMPLES
         =========
         >> Variable(3, [1]) < 3
         False

         >>> X = Variable(3, 1)
         >>> Y = Variable(4, [1])
         >>> X < Y
         True
        """
        try:
            return self.var < other.var
        except AttributeError:
            return self.var < other

    def __le__(self, other):
        """Dunder method for less than or equal to
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable

         RETURNS
         ========
         output (bool): True/False, less than or equal to/ not less than or equal to.

         EXAMPLES
         =========
         >> Variable(3, [1]) <= 3
         True

         >>> X = Variable(3, 1)
         >>> Y = Variable(4, [1])
         >>> X <= Y
         True
        """
        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= other

    def __gt__(self, other):
        """Dunder method for greater than
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable

         RETURNS
         ========
         output (bool): True/False, greater than/ not greater than.

         EXAMPLES
         =========
         >> Variable(3, [1]) > 3
         False

         >>> X = Variable(3, 1)
         >>> Y = Variable(4, [1])
         >>> X > Y
         False
        """
        try:
            return self.var > other.var
        except AttributeError:
            return self.var > other

    def __ge__(self, other):
        """Dunder method for greater than or equal to
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable

         RETURNS
         ========
         output (bool): True/False, greater than or equal to/ not greater than or equal to.

         EXAMPLES
         =========
         >>> Variable(3, [1]) >= 3
         True

         >>> X = Variable(3, 1)
         >>> Y = Variable(4, [1])
         >>> X >= Y
         False
        """
        try:
            return self.var >= other.var
        except AttributeError:
            return self.var >= other

    def __abs__(self):
        """Dunder method for absolute value
         INPUTS
         =======
         self: Variable object

         RETURNS
         ========
         output: Variable object after taking the absolute value

         EXAMPLES
         =========
         >>> abs(ariable(-3, [-1]))
         Variable(3, [1])
        """
        f = Reverse_Mode(abs(self.var))
        multiplier = 1
        if self.var < 0 :
            multiplier = -1
        der = self.var / abs(self.var) * multiplier
        self.child.append((der, f))
        return f

    def __pow__(self, exponent):
        """Returns the power of the Variable object to the exponent.
         INPUTS
         =======
         self: Variable object
         exponent: int/float, to the power of

         RETURNS
         ========
         power: a new Variable object after raising `self` to the power of `exponent`

         NOTES
         =====
         currently the base has to be > 0 and exponent has to be >= 1.
         complex numbers not supported
         EXAMPLES
         =========
         >>> x = Variable(3, 1)
         >>> x ** 2.0
         Variable(9, [6])
         """
        # `self` ^ other
        # check domain
        if self.var <= 0 and exponent < 1:
            raise ValueError('Base has to be > 0, and the exponent has to be >= 1')

        f = Reverse_Mode(self.var ** exponent)
        der = 1 * exponent * (self.var ** (exponent - 1))
        self.child.append((der, f))
        return f

    def __rpow__(self, other):
        """Returns the power of the `other` object to `self`.
         INPUTS
         =======
         self: Variable object, to the power of
         other: int/float, base

         RETURNS
         ========
         power: a new Variable object after raising `other` to the power of `self`
         NOTES
         =====
         currently do not support when `self` and `other` are both Variable type

         EXAMPLES
         =========
         >>> x = Variable(3, 1)
         >>> 2.0 ** x
         Variable(8, [5.545])
         """
        # `other` ^ `self`
        if other < 0 and self.var < 1:
            raise ValueError('Please input a non-negative value for the base. The exponent has to be >= 1')
        f = Reverse_Mode(other ** self.var)
        der = (other ** self.var) * np.log(other) * 1
        self.child.append((der, f))
        return f

    @staticmethod
    def sqrt(variable):
        """Returns the square root of `variable`.
         INPUTS
         =======
         variable: Variable object/int/float

         RETURNS
         ========
         sqrt: a new Variable object after taking square root of `variable`

         EXAMPLES
         =========
         >>> x = Variable(3)
         >>> Variable.sqrt(x)
         Variable(1.732, [0.289])
         """
        if variable < 0:
            raise ValueError('Cannot take sqrt of a negative value')
        return variable ** (1 / 2)

    @staticmethod
    def exp(variable):
        """Returns e to the value.
         INPUTS
         =======
         variable: Variable object/int/float

         RETURNS
         ========
         sqrt: a new Variable object after raising e to the value

         EXAMPLES
         =========
         >>> x = Variable(5)
         >>> Variable.exp(x)
         Variable(1.732, [0.289])
         """
        try:
            f = Reverse_Mode(np.exp(variable.var))
            variable.child.append((np.exp(variable.var), f))
        except AttributeError:
            f = np.exp(variable)

        return f

    @staticmethod
    def log(variable):
        """Returns the natural log of `variable`.
         INPUTS
         =======
         variable: Variable object/int/float

         RETURNS
         ========
         sqrt: a new Variable object after taking natural log

         EXAMPLES
         =========
         >>> x = Variable(3)
         >>> Variable.log(x)
         Variable(1.732, [0.289])
         """
        if variable <= 0:
            raise ValueError('Please input a positive number')
        try:
            f = Reverse_Mode(np.log(variable.var))
            der = (1.0 / variable.var) * 1
            variable.child.append((der, f))
            return f
        except AttributeError:
            return np.log(variable)

    @staticmethod
    def sin(variable):
        try:
            f = Reverse_Mode(np.sin(variable.var))
            variable.child.append((np.cos(variable.var), f))
        except AttributeError:
            f = np.sin(variable)

        return f

    @staticmethod
    def cos(variable):
        try:
            f = Reverse_Mode(np.cos(variable.var))
            variable.child.append((-np.sin(variable.var), f))
        except AttributeError:
            f = np.cos(variable)
        return f

    @staticmethod
    def tan(variable):
        """Returns the tangent of the Variable object.
        INPUTS
        =======
        variable: Variable object/int/float

        Returns
        =========
        output: a new Variable object after taking the tangent

        EXAMPLES
        =========
        >>> x = Variable(np.pi)
        >>> Variable.tan(x)
        Variable(-1.22464679915e-16, [ 1.])
        """

        # need to check that self.var is not a multiple of pi/2 + (pi * n), where n is a positive integer
        # would typically do try-except, but due to machine precision this won't work
        try:
            check_domain = variable.var % np.pi == (np.pi / 2)
            if check_domain:
                raise ValueError(
                    'Cannot take the tangent of this value since it is a multiple of pi/2 + (pi * n), where n is a positive integer')

            new_var = np.tan(variable.var)
            f = Reverse_Mode(new_var)

            der = 1 / np.power(np.cos(variable.var), 2) * 1
            variable.child.append((der, f))

            return f

        except AttributeError:
            return np.tan(variable)

    @staticmethod
    def arcsin(variable):
        """
        Returns the arcsine of Var object.

        INPUTS
        ==========
        variable: Variable object/int/float

        Returns
        =========
        output: a new Variable object after taking the arcsine

        Examples
        =========
        >>> x = Variable(0)
        >>> Variable.arcsin(x)
        Variable(0.0, [1.])
        """
        try:
            if variable.var > 1 or variable.var < -1:
                raise ValueError('Please input -1 <= x <=1')

            else:
                f = Reverse_Mode(np.arcsin(variable.var))
                der = 1 / np.sqrt(1 - (variable.var ** 2))
                variable.child.append((der, f))

            return f
        except AttributeError:
            return np.arcsin(variable)

    @staticmethod
    def arccos(variable):
        """
        Returns the arccosine of Var object.

        INPUTS
        ==========
        variable: Variable object/int/float

        Returns
        =========
        output: a new Variable object after taking the arccosine

        Examples
        =========
        >>> x = Variable(0)
        >>> Variable.arccos(x)
        Variable(0.0, [-1.])
        """
        try:
            if variable.var > 1 or variable.var < -1:
                raise ValueError('Please input -1 <= x <=1')

            else:
                f = Reverse_Mode(np.arcsin(variable.var))
                der = -1 / np.sqrt(1 - (variable.var ** 2))
                variable.child.append((der, f))

            return f
        except AttributeError:
            return np.arccos(variable)

    @staticmethod
    def arctan(variable):
        """Returns the arctangent of the Variable object.
        INPUTS
        =======
        variable: Variable object/int/float

        Returns
        =========
        output: a new Variable object after taking the arctangent

        EXAMPLES
        =========
        >>> x = Variable(np.pi)
        >>> Variable.arctan(x)
        Variable(1.26262725568, [ 0.09199967])
        """

        # no need to check for a value error
        try:
            f = Reverse_Mode(np.arctan(variable.var))
            der = 1 / (1 + np.power(variable.var, 2)) * 1
            variable.child.append((der, f))
            return f

        except AttributeError:
            return np.arctan(variable)

    @staticmethod
    def sinh(variable):
        """Returns the hyperbolic sin of the Variable object.

        INPUTS
        =======
        variable: Variable object/int/float

        Returns
        =========
        output: a new Variable object after taking the hyperbolic sine


        EXAMPLES
        =========
        >>> x = Variable(1)
        >>> Variable.sinh(x)
        Variable(1.17520119364, [ 1.54308063])
        """

        # don't need to check for domain values
        try:
            f = Reverse_Mode(np.sinh(variable.var))
            der = np.cosh(variable.var) * 1
            variable.child.append((der, f))
            return f

        except AttributeError:
            return np.sinh(variable)

    @staticmethod
    def cosh(variable):
        """Returns the hyperbolic cosine of the Variable object.

        INPUTS
        =======
        variable: Variable object/int/float

        Returns
        =========
        output: a new Variable object after taking the hyperbolic cosine

        EXAMPLES
        =========
        >>> x = Variable(1)
        >>> Variable.cosh(x)
        Variable(1.54308063482, [ 1.17520119])
        """

        # don't need to check for domain values

        try:
            f = Reverse_Mode(np.cosh(variable.var))
            der = np.sinh(variable.var) * 1
            variable.child.append((der, f))
            return f

        except AttributeError:
            return np.cosh(variable)

    @staticmethod
    def tanh(variable):
        """Returns the hyperbolic tangent of the Reverse_Mode object.

        INPUTS
        =======
        self: a Reverse_Mode object

        Returns
        =========
        output: a new Reverse_Mode object after the operation

        EXAMPLES
        =========
        >>> x = Variable(1)
        >>> Variable.tanh(x)
        Variable(0.761594155956 , [ 0.41997434])
        """

        # don't need to check for domain values
        try:
            f = Reverse_Mode(np.tanh(variable.var))
            der = 1 / np.power(np.cosh(variable.var), 2)
            variable.child.append((der, f))
        except AttributeError:
            f = np.tanh(variable)

        return f

    def __repr__(self):
        return 'Value: ' + str(self.var) + ' , Der: ' + str(self.der)

    def __str__(self):
        return 'Value: ' + str(self.var) + ' , Der: ' + str(self.der)

# x = Reverse_Mode(1)
# y = Reverse_Mode(2)
# inputs = [x, y]
# f1 = x*y+Reverse_Mode.exp(x*y)
# f2 = x + 3 * y
# value, check = f.derivative(inputs)

# x = Reverse_Mode(1)
# y = Reverse_Mode(2)
# ind_vars = [x, y]
# f = x*y+Reverse_Mode.exp(x*y)
# # for i in ind_vars:
# #     print(i.grad())
# check = f.derivative([x, y])
# print(check[0])  # [16.7781122  8.3890561]
# #cProfile.run('f.derivative([x,y])')


# HAVING ISSUES WITH NEGATION/SUBTRACTION
# x = Reverse_Mode(2)
# y = Reverse_Mode(3)

# b = x-4
# f = b-y
# print(f.var)
# print(y.grad())


'''
MY CONCLUSION IS THAT IF YOU DON'T HAVE A CHILD NODE, THEN YOU'RE THE FUNCTION
BASICALLY JUST NEED TO GET THE WEIGHT FOR THE SPECIFIC OPERATION, AND THEN CALL RECURSION UNTIL I GET TO REVERSE_MODE INSTANCE THAT HAS NO CHILDREN

    EXAMPLES
    =========
    # multiple functions
    >> x = Variable(3, [1, 0])
    >> y = Variable(4, [0, 1])
    >> f1 =  Variable.cos(x) + y ** 2
    >> f2 = 2 * Variable.log(y) - Variable.sqrt(x)/3 
    >> z = Functions([f1, f2])
    Values: [XXX, XXX]
    Derivative: 
    [ [ xxx, xxx],
    [ xxx, xxx] ]


With forward mode, you only have to call everything once and don't have to store anything
With reverse mode, need to store the input variables

x = Reverse_Mode(1)
y = Reverse_Mode(2)
f = x * y + Variable.exp(x*y)
f.derivative(seed = 1)


I'm currently recomputing things in this

'''

class Functions:
    def __init__(self, function, inputs):
        #         """Initiate a function variable.
        #          INPUTS
        #          =======
        #          self: Variable object
        #          var: float/int, the value of this variable
        #          seed: int/list/array, the seed vector (derivative from the parents)
        #          RETURNS
        #          ========
        #          EXAMPLES
        #          =========
        #          # multiple functions
        #          >> x = Variable(3, [1, 0])
        #          >> y = Variable(4, [0, 1])
        #          >> f1 =  Variable.cos(x) + y ** 2
        #          >> f2 = 2 * Variable.log(y) - Variable.sqrt(x)/3
        #          >> z = Functions([f1, f2])
        #          Values: [XXX, XXX]
        #          Derivative:
        #          [ [ xxx, xxx],
        #            [ xxx, xxx] ]

        #          """
        #         # if isinstance(function, list):
        #         #     check = [1 if isinstance(i, Variable) else 0 for i in var]
        #         #     if len(check) != sum(check):
        #         #         raise TypeError('Each function should be a variable class')
        #         # else:
        #         #     raise TypeError(
        #         #         'The input should be a list of variables standing for functions')
        self.var = np.array([i.var for i in function])
        # this because I'm passsing in a list, we need to do a deepcopy of the list
        # print(inputs) #
        # deep_copy_inputs = []
        # for _ in range(len(function)):
        #     new_input = copy.deepcopy(inputs)
        #     deep_copy_inputs.append(new_input)
        # for func, i in zip(function, deep_copy_inputs):
        #     print(func, i) # [Value: 1 , Der: None, Value: 2 , Der: None], [Value: 1 , Der: None, Value: 2 , Der: None]]
        # for func, inp in zip(function, deep_copy_inputs):
        #     print(func.derivative(inp[0]))
        # deep_copy_inputs = [copy.deepcopy(inputs) for _ in range(len(function))]
        # print(deep_copy_inputs) # [[Value: 1 , Der: None, Value: 2 , Der: None]]
        self.der = [list(func.derivative(inputs)[1]) for func in function]

if __name__ == "__main__":
    x = Reverse_Mode(1)
    y = Reverse_Mode(2)
    inputs = [x, y]
    f1 = x * y + Reverse_Mode.exp(x * y)
    f2 = x + 3 * y
    fcts = [f1, f2]
    f = Functions(fcts, inputs)
    print(f.der)