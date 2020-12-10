import numpy as np


class Reverse_Mode:

    def __init__(self, var):
        """Initiate a auto diff variable in the Reverse Mode.
          INPUTS
          =======
          self: Variable object
          var: float/int, the value of this variable

          RETURNS
          ========

          EXAMPLES
          =========
          # single var with der int
          >>> x = Reverse_Mode(3)
          """

        if isinstance(var, (int, float)):
            self.var = var
        else:
            raise TypeError('You did not enter a valid integer or float.')

        # need this because we need to store a form of the computational graph
        self.child = []
        self.der = None

    def gradient(self):
        """Calculate the gradient down the computation graph recursively. Should not be called seperately. Used inside
            self.derivative(*args)

          INPUTS
          =======
          self: Reverse_Mode object

          RETURNS
          ========
          self.der: the gradient of the current variable
          """

        if self.der is None:
            # I'm using this based on the equation for lecture 12 we had in class
            df_dui = 0
            for duj_dui, df_duj in self.child:
                # I need this += so that I can handle when the node has more than one child
                df_dui += duj_dui * df_duj.gradient()
            self.der = df_dui
        return self.der

    def derivative(self, inputs, seed=1):
        """Calcuate the gradients from the final function respect to each input variable.

          INPUTS
          =======
          self: Reverse_Mode object
          inputs: Reverse_Mode, each individual variable in the function

          RETURNS
          ========
          value: the value of the computation
          derivative: a Numpy array that contains the derivative respect to each variable

          EXAMPLES
          =========
          # single var with der int
          >>> x = Reverse_Mode(3), y = Reverse_Mode(4)
          >>> f = x * y
          >>> inputs = [x, y]
          >>> f.derivative(inputs)
          12, np.array([4, 3])
          """

        self.der = seed
        value = self.var
        derivatives = np.array([i.gradient() for i in inputs])
       # self._reset(inputs)

        return value, derivatives

    # @staticmethod
    # def _reset(inputs):
    #     for i in inputs:
    #         i.der = None

    def __add__(self, other):
        """Dunder method for adding another Reverse_Mode object or scalar/vector
          INPUTS
          =======
          self: Reverse_Mode object
          other: int/float/Reverse_Mode, to add on self.var.

          RETURNS
          ========
          output: Reverse_Mode, a new variable with `other` added, also computational graph updated.

          EXAMPLES
          =========
          add scalar
          >>> x = Reverse_Mode(3)
          >>> x + 3
          Reverse_Mode(6)

          add another variable - X
          >>> x = Reverse_Mode(3)
          >>> y = Reverse_Mode(5)
          >>> x + y
          Reverse_Mode(8)
         """

        try:
            f = Reverse_Mode(self.var + other.var)
            other.child.append((1.0, f))
            self.child.append((1.0, f))
        except AttributeError:
            f = Reverse_Mode(self.var + other)
            self.child.append((1.0, f))
        return f

    def __radd__(self, other):
        """Dunder method for adding another variable or scalar/vector from left
           INPUTS
           =======
           self: Reverse_Mode
           other: int/float

           RETURNS
           ========
           output: Reverse_Mode, a new variable with `other` added.

           EXAMPLES
           =========
           add scalar
           >>> x = Reverse_Mode(3)
           >>> 3 + x
           Variable(6)
          """

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
         >>> x = Reverse_Mode(4)
         >>> -x
         Reverse_Mode(-4)
        """

        f = Reverse_Mode(-self.var)
        self.child.append((-1, f))
        return f

    def __mul__(self, other):
        """Dunder method for multiplying another variable or scalar/vector
          INPUTS
          =======
          self: Reverse_Mode object
          other: int/float/Reverse_Mode, to multiply on self.var.

          RETURNS
          ========
          output: Reverse_Mode, a new variable with `other` multiplied.

          EXAMPLES
          =========
          multiplies scalar
          >>> x = Reverse_Mode(3)
          >>> x * 3
          Reverse_Mode(9)

          multiplies another variable - Y
          >> x = Reverse_Mode(3)
          >> y = Reverse_Mode(5)
          >> x * y
          Reverse_Mode(15)
         """

        try:
            f = Reverse_Mode(self.var * other.var)
            self.child.append((other.var, f))
            other.child.append((self.var, f))
        except AttributeError:
            f = Reverse_Mode(self.var * other)
            self.child.append((other, f))
        return f

    def __rmul__(self, other):
        """Dunder method for multiplying another variable or scalar from left
         INPUTS
         =======
         self: Reverse_Mode
         other: int/float

         RETURNS
         ========
         output: Reverse_Mode, a new variable with `other` multiplied.

         EXAMPLES
         =========
         multiplies scalar
         >>> x = Reverse_Mode(3)
         >>> 3 * x
         Reverse_Mode(9)
        """

        return self.__mul__(other)

    def __sub__(self, other):
        """Dunder method for subtracting another variable or scalar/vector
           INPUTS
           =======
           self: Reverse_Mode object
           other: int/float/Reverse_Mode, to subtract from self.var.

           RETURNS
           ========
           output: Reverse_Mode, a new variable that self.var subtracts `other`.

           EXAMPLES
           =========
           subtracts scalar
           >>> x = Reverse_Mode(10)
           >>> x - 3
           Reverse_Mode(7)

           multiplies another variable - X
           >>> x = Reverse_Mode(3, [1])
           >>> y = Reverse_Mode(2, [1])
           >>> x * y
           Reverse_Mode(6)
          """

        return self.__add__(-other)

    def __rsub__(self, other):
        """Dunder method for subtracted by another variable or scalar/vector
          INPUTS
          =======
          self: Variable object
          other: int/float, to subtract on self.var.

          RETURNS
          ========
          var: Reverse_Mode, a new variable that `other` subtracts self.var.

          EXAMPLES
          =========
          subtracted by scalar
          >>> x = Reverse_Mode(3, [1])
          >>> 3 - x
          Reverse_Mode(0)

          subtracted by another variable - X
          >>> x = Reverse_Mode(3)
          >>> y = Reverse_Mode(2)
          >>> x - y
          Reverse_Mode(1)
         """

        return (-self).__add__(other)

    def __truediv__(self, other):
        """Dunder method for dividing another variable or scalar/vector
         INPUTS
         =======
         self: Reverse_Mode object
         other: int/float/Reverse_Mode, to divide self.var.

         RETURNS
         ========
         output: Reverse_Mode, a new variable that self.var divides `other`.

         EXAMPLES
         =========
         divides scalar
         >>> x = Reverse_Mode(3)
         >>> x / 3
         Reverse_Mode(1)

         divides another variable - Y
         >>> x = Reverse_Mode(3)
         >>> y = Reverse_Mode(1)
         >>> x / y
         Reverse_Mode(3)
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
         self: Reverse_Mode object
         other: int/float/Reverse_Mode, to be divided by self.var.

         RETURNS
         ========
         output: Reverse_Mode, a new variable 'other' divides self.var.

         EXAMPLES
         =========
         divided by scalar
         >>> x = Reverse_Mode(3, [1])
         >>> 2 / x
         Reverse_Mode(2/3)
        """

        f = Reverse_Mode(other / self.var)
        der = other * (-self.var ** (-2)) * 1
        self.child.append((der, f))

        return f

    def __eq__(self, other):
        """Dunder method for checking equality
         INPUTS
         =======
         self: Reverse_Mode object
         other: int/float/Reverse_Mode, to be checked if it is equal

         RETURNS
         ========
         output: bool, True/False, Equal/Not Equal.

         EXAMPLES
         =========
         >>> Reverse_Mode(3) == 3
         False

         >>> X = Reverse_Mode(3)
         >>> Y = Reverse_Mode(3)
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
        """Dunder method for checking inequality. equality means self.var == other.var/other
         INPUTS
         =======
         self: Reverse_Mode object
         other: int/float/Reverse_Mode, to be checked if it is not equal

         RETURNS
         ========
         output: bool, True/False, Not Equal/ Equal.

         EXAMPLES
         =========
         >> Reverse_Mode(3) == 3
         True

         >>> X = Reverse_Mode(4)
         >>> Y = Reverse_Mode(3)
         >>> Y != X
         False
        """

        return not self.__eq__(other)

    def __lt__(self, other):
        """Dunder method for less than
         INPUTS
         =======
         self: Reverse_Mode object
         other: int/float/Reverse_Mode

         RETURNS
         ========
         out: bool, True/False, less than/ not less than.

         EXAMPLES
         =========
         >> Reverse_Mode(3) < 3
         False

         >>> X = Reverse_Mode(3)
         >>> Y = Reverse_Mode(4)
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
         self: Reverse_Mode object
         other: int/float/Reverse_Mode

         RETURNS
         ========
         output (bool): True/False, less than or equal to/ not less than or equal to.

         EXAMPLES
         =========
         >> Reverse_Mode(3) <= 3
         True

         >>> X = Reverse_Mode(5)
         >>> Y = Reverse_Mode(4)
         >>> X <= Y
         False
        """

        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= other

    def __gt__(self, other):
        """Dunder method for greater than
         INPUTS
         =======
         self: Reverse_Mode object
         other: int/float/Reverse_Mode

         RETURNS
         ========
         output (bool): True/False, greater than/ not greater than.

         EXAMPLES
         =========
         >> Reverse_Mode(3) > 3
         False

         >>> X = Reverse_Mode(5)
         >>> Y = Reverse_Mode(4)
         >>> X > Y
         True
        """

        try:
            return self.var > other.var
        except AttributeError:
            return self.var > other

    def __ge__(self, other):
        """Dunder method for greater than or equal to
         INPUTS
         =======
         self: Reverse_Mode object
         other: int/float/Reverse_Mode

         RETURNS
         ========
         output (bool): True/False, greater than or equal to/ not greater than or equal to.

         EXAMPLES
         =========
         >>> Reverse_Mode(3) >= 3
         True

         >>> X = Reverse_Mode(3)
         >>> Y = Reverse_Mode(4)
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
         self: Reverse_Mode object

         RETURNS
         ========
         output: Reverse_Mode object after taking the absolute value

         EXAMPLES
         =========
         >>> abs(Reverse_Mode(-3))
         Reverse_Mode(3)
        """

        f = Reverse_Mode(abs(self.var))
        multiplier = 1
        if self.var < 0 :
            multiplier = -1
        der = self.var / abs(self.var) * multiplier
        self.child.append((der, f))
        return f

    def __pow__(self, exponent):
        """Returns the power of the Reverse_Mode object to the exponent.
         INPUTS
         =======
         self: Reverse_Mode object
         exponent: int/float, to the power of

         RETURNS
         ========
         power: a new Reverse_Mode object after raising `self` to the power of `exponent`

         NOTES
         =====
         currently the base has to be > 0 and exponent has to be >= 1.
         complex numbers not supported. Currently not support base and exponent are Reverse_Mode objects

         EXAMPLES
         =========
         >>> x = Reverse_Mode(3)
         >>> x ** 2
         Reverse_Mode(9)
         """

        # `self` ^ other
        # check domain
        if self.var <= 0 and exponent < 1:
            raise ValueError('Base has to be > 0, and the exponent has to be >= 1')

        if isinstance(exponent, Reverse_Mode):
            raise AttributeError('The exponent cannot be a Reverse_Mode object')

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
        """Returns the square root of some value.
         INPUTS
         =======
         variable: Reverse_Mode object/int/float

         RETURNS
         ========
         sqrt: a new Reverse_Mode/int/float object after taking square root of `variable`

         EXAMPLES
         =========
         >>> x = Reverse_Mode(3)
         >>> Reverse_Mode.sqrt(x)
         Reverse_Mode(1.732)

         >>> Reverse_Mode.sqrt(3)
         1.732
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
         >>> x = Reverse_Mode(5)
         >>> Reverse_Mode.exp(x)
         Reverse_Mode(1.732)

         >>> Reverse_Mode.exp(5)
         1.732
         """

        try:
            f = Reverse_Mode(np.exp(variable.var))
            variable.child.append((np.exp(variable.var), f))
        except AttributeError:
            f = np.exp(variable)

        return f

    @staticmethod
    def log(variable):
        """Returns the natural log of some value.
         INPUTS
         =======
         variable: Reverse_Mode object/int/float

         RETURNS
         ========
         sqrt: a new Reverse_Mode object after taking natural log

         EXAMPLES
         =========
         >>> x = Reverse_Mode(3)
         >>> Reverse_Mode.log(x)
         Variable(1.732)

         >>> Reverse_Mode.log(3)
         1.732
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
        """Returns the sine of variable.

         INPUTS
         ==========
         variable: Reverse_Mode object/int/float

         Returns
         =========
         output: a new Reverse_Mode/int/float object after taking the sine

         Examples
         >>> x = Reverse_Mode(0)
         >>> Reverse_Mode.sin(x)
         Reverse_Mode(0.0)
         =========
         """

        try:
            f = Reverse_Mode(np.sin(variable.var))
            variable.child.append((np.cos(variable.var), f))
        except AttributeError:
            f = np.sin(variable)

        return f

    @staticmethod
    def cos(variable):
        """Returns the sine of variable.

         INPUTS
         ==========
         variable: Reverse_Mode object/int/float

         Returns
         =========
         output: a new Reverse_Mode/int/float object after taking the sine

         Examples
         >>> x = Reverse_Mode(0)
         >>> Reverse_Mode.sin(x)
         Reverse_Mode(0.0)
         =========
         """
        try:
            f = Reverse_Mode(np.cos(variable.var))
            variable.child.append((-np.sin(variable.var), f))
        except AttributeError:
            f = np.cos(variable)
        return f

    @staticmethod
    def tan(variable):
        """Returns the tangent of the variable.
        INPUTS
        =======
        variable: Reverse_Mode object/int/float

        Returns
        =========
        output: a new Reverse_Mode object after taking the tangent

        EXAMPLES
        =========
        >>> x = Reverse_Mode(np.pi)
        >>> Reverse_Mode.tan(x)
        Reverse_Mode(-1.22464679915e-16)
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
        Returns the arcsine of variable.

        INPUTS
        ==========
        variable: Reverse_Mode object/int/float

        Returns
        =========
        output: a new Reverse_Mode object after taking the arcsine

        Examples
        =========
        >>> x = Reverse_Mode(0)
        >>> Reverse_Mode.arcsin(x)
        Reverse_Mode(0.0)
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
        Returns the arccosine of variable.

        INPUTS
        ==========
        variable: Reverse_Mode object/int/float

        Returns
        =========
        output: a new Reverse_Mode object after taking the arccosine

        Examples
        =========
        >>> x = Reverse_Mode(0)
        >>> Reverse_Mode.arccos(x)
        Reverse_Mode(0.0)
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
        """Returns the arctangent of the variable.
        INPUTS
        =======
        variable: Reverse_Mode object/int/float

        Returns
        =========
        output: a new Reverse_Mode/inft/float object after taking the arctangent

        EXAMPLES
        =========
        >>> x = Reverse_Mode(np.pi)
        >>> Reverse_Mode.arctan(x)
        Reverse_Mode(1.26262725568)
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
        """Returns the hyperbolic sin of the variable.

        INPUTS
        =======
        variable: Reverse_Mode object/int/float

        Returns
        =========
        output: a new Reverse_Mode object/int/float after taking the hyperbolic sine


        EXAMPLES
        =========
        >>> x = Reverse_Mode(1)
        >>> Reverse_Mode.sinh(x)
        Reverse_Mode(1.17520119364)
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
        """Returns the hyperbolic cosine of the Variable.

        INPUTS
        =======
        variable: Variable object/int/float

        Returns
        =========
        output: a new Reverse_Mode object/int/float after taking the hyperbolic cosine

        EXAMPLES
        =========
        >>> x = Reverse_Mode(1)
        >>> Reverse_Mode.cosh(x)
        Reverse_Mode(1.54308063482)
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
        >>> x = Reverse_Mode(1)
        >>> Reverse_Mode.tanh(x)
        Reverse_Mode(0.761594155956)
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


class Functions:
    def __init__(self, function, inputs):

        self.der = []
        self.var = []
        for func in function:
            # for each function you're setting the Reverse Mode input variables
            ind_vars = []
            for i in inputs:
                ind_vars.append(Reverse_Mode(i))

            # how do you update the func so that it looks like ['ind_vars[0]*ind_vars[1]+Reverse_Mode.exp(ind_vars[0]*ind_vars[1])','ind_vars[0] + 3 * ind_vars[1]']?


            # calling eval on the string function
            f1 = eval(func)
            var, der = f1.derivative(ind_vars)
            # update the two attributes
            self.var.append(var)
            self.der.append(der)

        self.der = np.array(self.der)
        self.var = np.array(self.var)

    def __repr__(self):
        return 'Values: \n' + str(self.var) + '\n Gradients: \n' + str(self.der)

    def __str__(self):
        return 'Values: \n' + str(self.var) + '\n Gradients: \n' + str(self.der)


if __name__ == "__main__":
    x = 1
    y = 2
    inputs = [x, y]
    # f1 = 'x * y + Reverse_Mode.exp(x * y)'
    # f2 = 'x + 3 * y'
    # fcts = [f1, f2]
    fcts = ['ind_vars[0]*ind_vars[1]+Reverse_Mode.exp(ind_vars[0]*ind_vars[1])',
            'ind_vars[0] + 3 * ind_vars[1]']
    f = Functions(fcts, inputs)
    print(f)
