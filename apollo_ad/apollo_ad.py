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
        """Initiate a auto diff variable.
         INPUTS
         =======
         self: Variable object
         var: float/int, the value of this variable
         seed: int/list/array, the seed vector (derivative from the parents)

         RETURNS
         ========

         EXAMPLES
         =========
         # single var with der int
         >>> x = Variable(3, 1)
         Variable(3, [1])

         # single var with der list
         >>> x = Variable(3, [1])
         Variable(3, [1])

         # multiple vars
         >>> x = Variable(3, [1, 0])
         Variable(3, [1, 0])

         """
        if isinstance(var, (int, float)):
            self.var = var
        else:
            raise TypeError('You did not enter a valid integer or float.')
        if isinstance(seed, (int, float)):
            seed = np.array([seed])
        elif isinstance(seed, list):
            seed = np.array(seed)
        self.der = seed

    def __add__(self, other):
        """Dunder method for adding another variable or scalar/vector
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to add on self.var.
        
         RETURNS
         ========
         output: Variable, a new variable with `other` added.
            
         EXAMPLES
         =========
         add scalar
         >>> x = Variable(3, [1])
         >>> x + 3
         Variable(6, [1])

         add another variable - X
         >>> x = Variable(3, [1])
         >>> x + Variable(3, [1])
         Variable(6, [2])

         add another variable - Y
         >>> x = Variable(3, [1, 0])
         >>> x + Variable(3, [0, 1])
         Variable(6, [1, 1])
        """
        try:
            return Variable(self.var + other.var, self.der + other.der)
        except AttributeError:
            return Variable(self.var + other, self.der)

    def __radd__(self, other):
        """Dunder method for adding another variable or scalar/vector from left
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to add on self.var.
        
         RETURNS
         ========
         output: Variable, a new variable with `other` added.
            
         EXAMPLES
         =========
         add scalar
         >>> x = Variable(3, [1])
         >>> 3 + x
         Variable(6, 1)

         add another variable - X
         >>> x = Variable(3, [1])
         >>> Variable(3, [1]) + x
         Variable(6, [2])

         add another variable - Y
         >>> x = Variable(3, [1, 0])
         >>> Variable(3, [0, 1]) + x
         Variable(6, [1, 1])
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Dunder method for multiplying another variable or scalar/vector
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to multiply on self.var.
        
         RETURNS
         ========
         output: Variable, a new variable with `other` multiplied.
            
         EXAMPLES
         =========
         multiplies scalar
         >>> x = Variable(3, [1])
         >>> x * 3
         Variable(9, [3])

         multiplies another variable - X
         >>> x = Variable(3, [1])
         >>> x * Variable(3, [1])
         Variable(9, [6])

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
        """Dunder method for multiplying another variable or scalar/vector from left
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to multiply on self.var.
        
         RETURNS
         ========
         output: Variable, a new variable with `other` multiplied.
            
         EXAMPLES
         =========
         multiplies scalar
         >>> x = Variable(3, [1])
         >>> 3 * x
         Variable(9, [3])

         multiplies another variable - X
         >>> x = Variable(3, [1])
         >>> Variable(3, [1]) * x
         Variable(9, [6])

         multiplies another variable - Y
         >>> x = Variable(3, [1, 0])
         >>> Variable(3, [0, 1]) * x
         Variable(9, [3, 3])

        """
        return self.__mul__(other)

    def __sub__(self, other):
        """Dunder method for subtracting another variable or scalar/vector
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to subtract from self.var.
        
         RETURNS
         ========
         output: Variable, a new variable that self.var subtracts `other`.
            
         EXAMPLES
         =========
         subtracts scalar
         >>> x = Variable(3, [1])
         >>> x - 3
         Variable(0, [1])
 
         multiplies another variable - X
         >>> x = Variable(3, [1])
         >>> x - Variable(3, [1])
         Variable(0, [0])

         multiplies another variable - Y
         >>> x = Variable(4, [1, 0])
         >>> x - Variable(3, [0, 1])
         Variable(1, [1, -1])
        """
        return self.__add__(-other)

    def __rsub__(self, other):
        """Dunder method for subtracted by another variable or scalar/vector 
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable, to subtract on self.var.
        
         RETURNS
         ========
         var: Variable, a new variable that `other` subtracts self.var.
            
         EXAMPLES
         =========
         subtracted by scalar
         >>> x = Variable(3, [1])
         >>> 3 - x
         Variable(0, [-1])
 
         subtracted by another variable - X
         >>> x = Variable(3, [1])
         >>> Variable(3, [1]) - x
         Variable(0, [0])

         subtracted by another variable - Y
         >>> x = Variable(3, [1, 0])
         >>> Variable(4, [0, 1]) - x
         Variable(1, [-1, 1])
        """
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
            return Variable(self.var / other.var, (self.der * other.var - self.var * other.der)/(other.var**2))
        except AttributeError:
            return Variable(self.var / other, self.der / other)

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
        try:
            return Variable(other.var / self.var, (other.der * self.var - other.var * self.der)/(self.var**2))
        except AttributeError:
            return Variable(other / self.var, other * (-self.var**(-2)) * self.der)

    def __neg__(self):
        """Dunder method for taking the negative
         INPUTS
         =======
         self: Variable object
         other: int/float/Variable
        
         RETURNS
         ========
         output: Variable, a new variable in negation
            
         EXAMPLES
         =========
         >>> x = Variable(3, [1])
         >>> -x
         Variable(-3, [-1])
        """
        return Variable(-self.var, -self.der)

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
            out = (self.var == other.var) and np.array_equal(self.der, other.der)
        except AttributeError:
            # a scalar is not equal to a variable
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
        var = abs(self.var)
        der = np.abs(self.der)
        return Variable(var, der) 

    def __pow__(self, exponent):
        """Returns the power of the Variable object to the exponent.
         INPUTS
         =======
         self: Variable object
         exponent: Variable/int/float, to the power of

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
        try:
            var = self.var ** exponent.var
            der = exponent.var * (self.var ** (exponent.var - 1)) * self.der + \
                (self.var ** exponent.var) * np.log(self.var) * exponent.der
            return Variable(var, der)
        except AttributeError:
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

    @staticmethod
    def sin(variable):

        """Returns the sine of Var object.
        
        INPUTS
        ==========
        variable: Variable object/int/float
        
        Returns
        ========= 
        output: a new Variable object after taking the sine
        
        Examples
        >>> x = Variable(0)
        >>> Variable.sin(x)
        Variable(0.0, [1.])     
        ========= 
        """

        try:
            var = np.sin(variable.var)
            b = np.cos(variable.var)
            der = variable.der * b
            return Variable(var, der)
        except AttributeError:
            return np.sin(variable)

        

    @staticmethod
    def cos(variable):
        """ 
        Returns the cosine of Var object.
        
        INPUTS
        ==========
        variable: Variable object/int/float
        
        Returns
        ========= 
        output: a new Variable object after taking the cosine
        
        Examples
        >>> x = Variable(0)
        >>> Variable.cos(x)
        Variable(1.0, [0.])     
        ========= 

        """
        try:
            var = np.cos(variable.var)
            b = -np.sin(variable.var)
            der = variable.der * b

            return Variable(var, der)
        except AttributeError:
            return np.cos(variable)

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
            check_domain = variable.var % np.pi == (np.pi/2)
            if check_domain:
                raise ValueError(
                    'Cannot take the tangent of this value since it is a multiple of pi/2 + (pi * n), where n is a positive integer')

            new_var = np.tan(variable.var)

            tan_derivative = 1 / np.power(np.cos(variable.var), 2)
            new_der = variable.der * tan_derivative

            tan = Variable(new_var, new_der)

            return tan

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
            if variable.var>1 or variable.var <-1:
                raise ValueError('Please input -1 <= x <=1')

            else:
                var = np.arcsin(variable.var)
                der = 1 / np.sqrt(1 - (variable.var ** 2))


            return Variable(var, der) 
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
            if variable.var>1 or variable.var <-1:
                raise ValueError('Please input -1 <= x <=1')

            else:
                var = np.arcsin(variable.var)
                der = -1 / np.sqrt(1 - (variable.var ** 2))
            return Variable(var, der)
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
            new_var = np.arctan(variable.var)

            arctan_derivative = 1 / (1 + np.power(variable.var, 2))
            new_der = variable.der * arctan_derivative

            arctan = Variable(new_var, new_der)

            return arctan

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
            new_var = np.sinh(variable.var)

            sinh_derivative = np.cosh(variable.var)
            new_der = variable.der * sinh_derivative

            sinh = Variable(new_var, new_der)

            return sinh

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
            new_var = np.cosh(variable.var)

            cosh_derivative = np.sinh(variable.var)
            new_der = variable.der * cosh_derivative

            cosh = Variable(new_var, new_der)

            return cosh

        except AttributeError:
            return np.cosh(variable)

    @staticmethod
    def tanh(variable):
        """Returns the hyperbolic tangent of the Variable object.

        INPUTS
        =======
        variable: Variable object/int/float
        
        Returns
        ========= 
        output: a new Variable object after taking the hyperbolic tangent

        EXAMPLES
        =========
        >>> x = Variable(1)
        >>> Variable.tanh(x)
        Variable(0.761594155956 , [ 0.41997434])
        """

        # don't need to check for domain values
        try:
            new_var = np.tanh(variable.var)

            tanh_derivative = 1 / np.power(np.cosh(variable.var), 2)
            new_der = variable.der * tanh_derivative

            tanh = Variable(new_var, new_der)
            return tanh

        except AttributeError:
            return np.tanh(variable)


    def __repr__(self):
        return 'Value: ' + str(self.var) + ' , Der: ' + str(self.der) 

    def __str__(self):
        return 'Value: ' + str(self.var) + ' , Der: ' + str(self.der) 
