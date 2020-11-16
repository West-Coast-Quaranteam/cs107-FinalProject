import numpy as np


class Variable:

    """Summary
    
    Attributes
    ----------
    der : TYPE
        Description
    var : TYPE
        Description
    """
    
    def __init__(self, var, seed = np.array([1])):
        """Summary
        
        Parameters
        ----------
        var : TYPE
            Description
        seed : int, optional
            Description
        """
        self.var = var
        if isinstance(seed, (int, float)):
            seed = np.array([seed])
        elif isinstance(seed, list):
            seed = np.array(seed)
        self.der = seed

    def __add__(self, other):
        """Summary
        
        Parameters
        ----------
        other : int/float, Variable
            Description
        
        Returns
        -------
        var: Variable
            Description

        Examples
        -------
        add scalar
        >> x = Variable(3, [1])
        >> x + 3
        Variable(6, 1)

        add another variable - X
        >> x = Variable(3, [1])
        >> x + Variable(3, [1])
        Variable(6, [2])

        add another variable - Y
        >> x = Variable(3, [1, 0])
        >> x + Variable(3, [0, 1])
        Variable(6, [1, 1])

        """
        try:
            return Variable(self.var + other.var, self.der + other.der)
        except AttributeError:
            return Variable(self.var + other, self.der)

    def __radd__(self, other):
        """Summary
        
        Parameters
        ----------
        other : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Summary
        
        Parameters
        ----------
        other : int/float, Variable
            Description
        
        Returns
        -------
        var: Variable
            Description

        Examples
        -------
        multiplies scalar
        >> x = Variable(3, [1])
        >> x * 3
        Variable(18, 1)

        multiplies another variable - X
        >> x = Variable(3, [1])
        >> x + Variable(3, [1])
        Variable(6, [2])

        multiplies another variable - Y
        >> x = Variable(3, [1, 0])
        >> x * Variable(3, [0, 1])
        Variable(9, [3, 3])

        """
        try:
            return Variable(self.var * other.var, self.var * other.der + self.der * other.var)
        except AttributeError:
            return Variable(self.var * other, self.der * other)

    def __rmul__(self, other):
        """Summary
        
        Parameters
        ----------
        other : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        return self.__mul__(other)

    def __sub__(self, other):
        """Summary
        
        Parameters
        ----------
        other : int/float, Variable
            Description
        
        Returns
        -------
        var: Variable
            Description

        Examples
        -------
        multiplies scalar
        >> x = Variable(3, [1])
        >> x * 3
        Variable(18, 1)

        multiplies another variable - X
        >> x = Variable(3, [1])
        >> x - Variable(3, [1])
        Variable(0, [0])

        multiplies another variable - Y
        >> x = Variable(4, [1, 0])
        >> x * Variable(3, [0, 1])
        Variable(1, [1, -1])
        """
        return self.__add__(-other)

    def __rsub__(self, other):
        """Summary
        
        Parameters
        ----------
        other : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        return self.__sub__(other)

    def __truediv__(self, other):
        """Summary
        
        Parameters
        ----------
        other : int/float, Variable
            Description
        
        Returns
        -------
        var: Variable
            Description

        Examples
        -------
        divides scalar
        >> x = Variable(3, [1])
        >> x / 3
        Variable(1, 1)

        divides another variable - X
        >> x = Variable(3, [1])
        >> x / Variable(4, [1])
        Variable(3/4, [1/16])

        multiplies another variable - Y
        >> x = Variable(3, [1, 0])
        >> x / Variable(4, [0, 1])
        Variable(3/4, [1/4, -3/16])

        """
        try:
            return Variable(self.var / other.var, (self.der * other.var - self.var * other.der)/(other.var**2))
        except AttributeError:
            return Variable(self.var / other, self.der / other)

    def __rtruediv__(self, other):
        """Summary
        
        Parameters
        ----------
        other : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        """
        return self.__truediv__(other)

    def __neg__(self):
        """Summary

        Examples
        -------
        >> x = Variable(3, [1])
        >> -x
        Variable(-3, [-1])
        """
        return Variable(-self.var, -self.der)

    def __eq__(self, other):
        """Summary
        
        Parameters
        ----------
        other : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Examples
        -------
        >> Variable(3, [1]) == 3
        False

        >> X = Variable(3, 1)
        >> Y = Variable(3, [1])
        >> X == Y
        True
        """
        try:
            return (self.var == other.var) and np.array_equal(self.der, other.der)
        except AttributeError:
            # a scalar is not equal to a variable
            return False


    def __ne__(self, other):
        """Summary
        
        Parameters
        ----------
        other : TYPE
            Description
        
        Returns
        -------
        TYPE
            Description
        
        Examples
        -------
        >> X = Variable(3, 1)
        >> Y = Variable(3, [1])
        >> X != Y
        False
        """
        return not self.__eq__(other)

    def __lt__(self, other):
        try:
            return self.var < other.var
        except AttributeError:
            return self.var < other

    def __le__(self, other):
        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= other

    def __gt__(self, other):
        try:
            return self.var > other.var
        except AttributeError:
            return self.var > other

    def __ge__(self, other):
        try:
            return self.var >= other.var
        except AttributeError:
            return self.var >= other

    def __abs__(self):
        var = abs(self.var)
        der = np.abs(self.der)
        return Variable(var, der) 

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
         currently do not support when `self` and `other` are both Variable type
         self has to be positive, and exponent has to be >= 1. Don't support imaginary numbers

         EXAMPLES
         =========
         >>> x = Variable(3, 1)
         >>> x ** 2.0
         Variable(9, [6])
         """
        # `self` ** other
        # check domain, current
        if self.var < 0 and exponent < 1:
            raise ValueError('Please input a non-negative value for the base. The exponent has to be >= 1')

        var = self.var ** exponent
        der = self.der * exponent * (self.var ** (exponent - 1))
        return Variable(var, der)

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
        var = other ** self.var
        der = (other ** self.var) * np.log(other) * self.der
        return Variable(var, der)

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
        return variable ** (1/2)

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
            var = np.exp(variable.var)
            der = np.exp(variable.var) * variable.der
            return Variable(var, der)

        except AttributeError:
            return np.exp(variable)

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
            var = np.log(variable.var)
            der = (1.0 / variable.var) * variable.der
            return Variable(var, der)
        except AttributeError:
            return np.log(variable)

    def sin(self):
        raise NotImplementedError

    def cos(self):
        raise NotImplementedError

    def tan(self):
        """Returns the tangent of the Variable object.

        INPUTS
        =======
        self: Variable object

        RETURNS
        ========
        tan: a new Variable object

        EXAMPLES
        =========
        >>> x = Variable(np.pi)
        >>> Variable.tan(x)
        Variable(-1.22464679915e-16, [ 1.])
        """

        # need to check that self.var is not a multiple of pi/2 + (pi * n), where n is a positive integer
        # would typically do try-except, but due to machine precision this won't work
        check_domain = self.var % np.pi == (np.pi/2)
        if check_domain:
            raise ValueError(
                'Cannot take the tangent of this value since it is a multiple of pi/2 + (pi * n), where n is a positive integer')

        new_var = np.tan(self.var)

        tan_derivative = 1 / np.power(np.cos(self.var), 2)
        new_der = self.der * tan_derivative

        tan = Variable(new_var, new_der)

        return tan

    def arcsin(self):
        raise NotImplementedError

    def arccos(self):
        raise NotImplementedError

    def arctan(self):
        """Returns the arctangent of the Variable object.

        INPUTS
        =======
        self: Variable object

        RETURNS
        ========
        arctan: a new Variable object

        EXAMPLES
        =========
        >>> x = Variable(np.pi)
        >>> Variable.arctan(x)
        Variable(1.26262725568, [ 0.09199967])
        """

        # no need to check for a value error

        new_var = np.arctan(self.var)

        arctan_derivative = 1 / (1 + np.power(self.var, 2))
        new_der = self.der * arctan_derivative

        arctan = Variable(new_var, new_der)

        return arctan

    def sinh(self):
        """Returns the hyperbolic sin of the Variable object.

        INPUTS
        =======
        self: Variable object

        RETURNS
        ========
        sinh: a new Variable object

        EXAMPLES
        =========
        >>> x = Variable(1)
        >>> Variable.sinh(x)
        Variable(1.17520119364, [ 1.54308063])
        """

        # don't need to check for domain values

        new_var = np.sinh(self.var)

        sinh_derivative = np.cosh(self.var)
        new_der = self.der * sinh_derivative

        sinh = Variable(new_var, new_der)

        return sinh

    def cosh(self):
        """Returns the hyperbolic cosine of the Variable object.

        INPUTS
        =======
        self: Variable object

        RETURNS
        ========
        cosh: a new Variable object

        EXAMPLES
        =========
        >>> x = Variable(1)
        >>> Variable.cosh(x)
        Variable(1.54308063482, [ 1.17520119])
        """

        # don't need to check for domain values

        new_var = np.cosh(self.var)

        cosh_derivative = np.sinh(self.var)
        new_der = self.der * cosh_derivative

        cosh = Variable(new_var, new_der)

        return cosh

    def tanh(self):
        """Returns the hyperbolic tangent of the Variable object.

        INPUTS
        =======
        self: Variable object

        RETURNS
        ========
        tanh: a new Variable object

        EXAMPLES
        =========
        >>> x = Variable(1)
        >>> Variable.tanh(x)
        Variable(0.761594155956 , [ 0.41997434])
        """

        # don't need to check for domain values

        new_var = np.tanh(self.var)

        tanh_derivative = 1 / np.power(np.cosh(self.var), 2)
        new_der = self.der * tanh_derivative

        tanh = Variable(new_var, new_der)

        return tanh

    def __repr__(self):
        return 'Value: ' + str(self.var) + ' , Der: ' + str(self.der) 

    def __str__(self):
        return 'Value: ' + str(self.var) + ' , Der: ' + str(self.der) 


if __name__ == "__main__":
    # x = Variable(0, 1)
    # print(Variable.log(x))

    x = Variable(np.pi)
    print(Variable.tan(x))
    print(Variable.arctan(x))

    x = Variable(1)
    print(Variable.sinh(x))
    print(Variable.cosh(x))
    print(Variable.tanh(x))
