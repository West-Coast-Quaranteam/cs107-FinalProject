
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
            return False
        try:
            var = np.log(variable.var)
            der = (1.0 / variable.var) * variable.der
            return Variable(var, der)
        except AttributeError:
            return np.log(variable)

    @staticmethod
    def sin(self):
        """ 
        Returns the sine of Var object.
        
        INPUTS
        ==========
        self: Var object
        
        Returns
        ========= 
        output: sine of self
        
        Examples
        ========= 

        """
        var = np.sin(self.var)
        if len(self.der.shape):
            b = np.cos(self.var)
            der = self.der * b
        else:
            der = None

        return Variable(var, der)

    def cos(self):
        """ 
        Returns the cosine of Var object.
        
        INPUTS
        ==========
        self: Var object
        
        Returns
        ========= 
        output: cosine of self
        
        Examples
        ========= 

        """
        var = np.cos(self.var)
        if len(self.der.shape):
            b = -np.sin(self.var)
            der = self.der * b
        else:
            der = None

        return Variable(var, der)



    def tan(self):
        raise NotImplementedError

    def arcsin(self):
        """ 
        Returns the arcsine of Var object.
        
        INPUTS
        ==========
        self: Var object
        
        Returns
        ========= 
        output:  arcsine of self
        
        Examples
        ========= 

        """
        if self.var>1 or self.var <-1:
            raise ValueError('Please input -1 <= x <=1')
            return False
        else:
            var = np.arcsin(self.var)
            der = 1 / np.sqrt(1 - (self.var ** 2))


        return Variable(var, der) 

    def arccos(self):
        """ 
        Returns the arccosine of Var object.
        
        INPUTS
        ==========
        self: Var object
        
        Returns
        ========= 
        output: arccosine of self
        
        Examples
        ========= 

        """

        if self.var>1 or self.var <-1:
            raise ValueError('Please input -1 <= x <=1')
            return False
        else:
            var = np.arcsin(self.var)
            der = -1 / np.sqrt(1 - (self.var ** 2))
        return Variable(var, der)



    def arctan(self):
        raise NotImplementedError

    def sinh(self):
        raise NotImplementedError

    def cosh(self):
        raise NotImplementedError

    def tanh(self):
        raise NotImplementedError

    def __repr__(self):
        return 'varue: ' + str(self.var) + ' , Der: ' + str(self.der) 

    def __str__(self):
        return 'varue: ' + str(self.var) + ' , Der: ' + str(self.der) 


if __name__ == "__main__":
    x = Variable(0)
    print(Variable.sin(x))
    print(Variable.cos(x))
    print(Variable.arcsin(x))
    print(Variable.arccos(x))
