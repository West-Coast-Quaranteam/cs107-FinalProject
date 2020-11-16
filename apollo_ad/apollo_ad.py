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
            return self.var < var

    def __le__(self, other):
        try:
            return self.var <= other.var
        except AttributeError:
            return self.var <= var

    def __gt__(self, other):
        try:
            return self.var > other.var
        except AttributeError:
            return self.var > var

    def __ge__(self, other):
        try:
            return self.var >= other.var
        except AttributeError:
            return self.var >= var

    def __abs__(self):
        var = abs(self.var)
        der = np.abs(self.der)
        return Variable(var, der) 

    def __pow__(self):
        raise NotImplementedError

    def __rpow__(self):
        raise NotImplementedError

    def sqrt(self):
        raise NotImplementedError

    def exp(self):
        raise NotImplementedError

    def log(self):
        raise NotImplementedError

    def sin(self):
        """ 
        Returns the sine of Var object.
        
        Parameters
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
            b = np.expand_dims(b, 1) if len(self.der.shape) > len(b.shape) else b
            der = self.der * b
        else:
            der = None

        return Variable(var, der)

    def cos(self):
        """ 
        Returns the cosine of Var object.
        
        Parameters
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
            b = np.expand_dims(b, 1) if len(self.der.shape) > len(b.shape) else b
            der = self.der * b
        else:
            der = None

        return Variable(var, der)



    def tan(self):
        raise NotImplementedError

    def arcsin(self):
        """ 
        Returns the arcsine of Var object.
        
        Parameters
        ==========
        self: Var object
        
        Returns
        ========= 
        output:  arcsine of self
        
        Examples
        ========= 

        """
        values = map(lambda x: -1 <= x <= 1, self.var)
        if not all(values):
            raise ValueError("varueError: domain of arcsin is [-1, 1].")        
        var = np.arcsin(self.var)
        if len(self.der.shape):
            if self.var == 1:
                b = np.nan
            elif self.var == -1:
                b = np.nan
            else:
                b = 1 / np.sqrt(1 - (self.var ** 2))
                b = np.expand_dims(b, 1) if len(self.der.shape) > len(b.shape) else b
            der = b * self.der
        else:
            der = None
        return Variable(var, der)    

    def arccos(self):
        """ 
        Returns the arccosine of Var object.
        
        Parameters
        ==========
        self: Var object
        
        Returns
        ========= 
        output: arccosine of self
        
        Examples
        ========= 

        """
        values = map(lambda x: -1 <= x <= 1, self.var)
        if not all(values):
            raise ValueError("varueError: domain of arccos is [-1, 1].")    
        var = np.arccos(self.var)
        if len(self.der.shape):
            if self.var == 1:
                b = np.nan
            elif self.var == -1:
                b = np.nan
            else:
                b = -1 / np.sqrt(1 - (self.var ** 2))
                b = np.expand_dims(b, 1) if len(self.der.shape) > len(b.shape) else b
            der = b * self.der
        else:
            der = None
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
    a = 2
    alpha = 2.0
    beta = 3.0

    x = Variable(a)
    f = alpha * x + beta
    print(f)
    assert f.var == 7.0 and f.der == [2.0]

    f = x * alpha + beta
    print(f)
    assert f.var == 7.0 and f.der == [2.0]

    f = beta + alpha * x
    print(f)
    assert f.var == 7.0 and f.der == [2.0]

    f = beta + x * alpha
    print(f)
    assert f.var == 7.0 and f.der == [2.0]

    x = Variable(3, [1])
    x = -x 
    print(x)
    assert x.var == -3 and x.der == [-1]

    x = Variable(3, [1])
    f = x - Variable(3, [1])
    print(f)
    assert f.var == 0  and f.der == [0]

    x = Variable(3, [1])
    f = Variable(3, [1]) - x
    print(f)
    assert f.var == 0  and f.der == [0]

    f = Variable(3, [1]) - 3
    print(f)
    assert f.var == 0  and f.der == [1]

    print(Variable(3, [1]) == 3)

    X = Variable(3, 1)
    Y = Variable(3, [1])
    print(X == Y)

    x = Variable(3, [1, 0])
    f = x * Variable(3, [0, 1])
    print(f)
    assert f.var == 9 and (f.der == [3, 3]).all()
