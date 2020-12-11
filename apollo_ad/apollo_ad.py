import numpy as np
import re


class auto_diff:
    def __init__(self, var, fct, seed = None):
        """Initiate a function variable.
         INPUTS
         =======
         self: Variable object
         var: float/int, the value of this variable
         seed: int/list/array, the seed vector (derivative from the parents)

         RETURNS
         ========

         EXAMPLES
         =========
         
         # forward mode
         vars = {'x': 0.5, 'y': 4}
         fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3', 'sqrt(x)/3', '3 * sinh(x) - 4 * arcsin(x) + 5']
         z = auto_diff(vars, fcts)
         -- Values -- 
         Function F1: 16.87758256189037
         Function F2: 2.5368864618442655
         Function F3: 0.23570226039551587
         Function F4: 4.468890814088047
         -- Gradients -- 
         Function F1: [-0.47942554  8.        ]
         Function F2: [-0.23570226  0.5       ]
         Function F3: [0.23570226 0.        ]
         Function F4: [-1.23592426 -4.61880215]

         # reverse mode
         vars = {'x': 3, 'y': 4, 'z': 5}
         fcts = ['cos(x) + y ** 2 + sqrt(z)']
         z = auto_diff(vars, fcts)
         -- Values -- 
         Function F1: 17.246075480899343
         -- Gradients -- 
         Function F1: [-0.14112001  8.          0.2236068 ]

         If we have non-default seeds, 
         (for reverse mode, it should be a list of ints for each 
         functions whereas for forward mode, it is a dictionary of 
         ints for each variable)

         vars = {'x': 3, 'y': 4}
         seeds = {'x': 1, 'y': 2}
         fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
         z = auto_diff(vars, fcts, seeds)
         -- Values -- 
         x: 15.010007503399555
         y: 2.1952384530501554
         -- Gradients -- 
         Function F1: [-0.14112001 16.        ]
         Function F2: [-0.09622504  1.        ]
         """

        forward_mode = True
        if len(var) < len(fct):
            print('# of variables < # of functions ====> automatically use the forward mode!')
        elif len(var) > len(fct):
            print('# of variables > # of functions ====> automatically use the reverse mode!')
            forward_mode = False
        else:
            print('# of variables = # of functions ====> automatically use the forward mode!')

        if forward_mode:
            if (seed is not None) and (not isinstance(seed, dict)):
                raise AttributeError('Forward mode requires the seed to be a dict with variable: seed.')
            out = Forward(var, fct, seed)
        else:
            if (seed is not None) and (not isinstance(seed, list)):
                raise AttributeError('Reverse mode requires the seed to be a list with the seed for each function.')
            out = Reverse(var, fct, seed)

        self.out = out
        self.var = out.var
        self.der = out.der

    def __repr__(self): 
        return self.out.__repr__()

    def __str__(self):
        return self.out.__str__()

class Forward:
    def __init__(self, var, fct, seed = None):
        """Initiate a function variable.
         INPUTS
         =======
         self: Variable object
         var: float/int, the value of this variable
         seed: int/list/array, the seed vector (derivative from the parents)

         RETURNS
         ========

         EXAMPLES
         =========

         vars = {'x': 3, 'y': 4}
         fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
         z = Forward(vars, fcts)
         -- Values -- 
         Function F1: 15.010007503399555
         Function F2: 3.2938507417182654
         -- Gradients -- 
         Function F1: [-0.14112001  8.        ]
         Function F2: [0.23710829 0.5       ]

         If we have non-default seed:

         vars = {'x': 3, 'y': 4}
         seeds = {'x': 1, 'y': 2}
         fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
         z = Forward(vars, fcts, seeds)
         -- Values -- 
         Function F1: 15.010007503399555
         Function F2: 2.1952384530501554
         -- Gradients -- 
         Function F1: [-0.14112001 16.        ]
         Function F2: [-0.09622504  1.        ]
         """

        check = [1 if isinstance(i, str) else 0 for i in fct]
        if len(check) != sum(check):
            raise TypeError('Each function should be a string!')
        else:
            if isinstance(var, dict):
                if seed is None:
                    # default seed
                    seed = {var_name: 1 for var_name in list(var.keys())}
                
                # initializing all variables
                num_var = len(var)
                self.var2idx = dict(zip(list(var.keys()), list(range(num_var))))
                for var_name, var_value in var.items():
                    der_ = np.zeros((num_var,))
                    der_[self.var2idx[var_name]] = float(seed[var_name])
                    exec(f'{var_name} = Variable(float(var_value), der_)')
                # initializing all functions
                
                # static methods
                static_methods = ['log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']

                all_fcts = []
                num_fct = len(fct)
                for function in fct:
                    for i in static_methods:
                        if i in function:
                            # for multiple but same static methods 
                            function = re.sub(i + r'\(', 'Variable.' + i + '(', function)
                            function = re.sub('arcVariable.', 'arc', function)
                    all_fcts.append(eval(function))

                self.var = np.array([i.var for i in all_fcts])
                self.der = np.array([i.der for i in all_fcts])

            else: 
                raise TypeError('The variable should be a dictionary!')

    
    def __repr__(self):
        output_string = '-- Values -- \n'
        for var_idx in range(self.var.shape[0]):
            output_string =  output_string + 'Function F' + str(var_idx + 1) + ': ' + str(self.var[var_idx]) + '\n'

        output_string = output_string + '-- Gradients -- \n'

        for fct_idx in range(self.der.shape[0]):
            output_string = output_string + 'Function F' + str(fct_idx + 1) + ': ' + str(self.der[fct_idx]) + '\n'

        return output_string

    def __str__(self):
        output_string = '-- Values -- \n'
        for var_idx in range(self.var.shape[0]):
            output_string =  output_string + 'Function F' + str(var_idx + 1) + ': ' + str(self.var[var_idx]) + '\n'

        output_string = output_string + '-- Gradients -- \n'

        for fct_idx in range(self.der.shape[0]):
            output_string = output_string + 'Function F' + str(fct_idx + 1) + ': ' + str(self.der[fct_idx]) + '\n'

        return output_string

class Reverse:
    def __init__(self, var, fct, seed = None):
        """Initiate a function variable.
         INPUTS
         =======
         self: Variable object
         var: float/int, the value of this variable
         seed: int/list/array, the seed vector (derivative from the parents)

         RETURNS
         ========

         EXAMPLES
         =========

         var = {'x': 1, 'y': 2}
         fcts = ['x * y + exp(x * y)', 'x + 3 * y']
         z = Reverse(var, fcts)
         -- Values -- 
         Function F1: 9.38905609893065
         Function F2: 7.0
         -- Gradients -- 
         Function F1: [16.7781122  8.3890561]
         Function F2: [1. 3.]

         If we have non-default seed for each function:

         var = {'x': 1, 'y': 2}
         fcts = ['x * y + exp(x * y)', 'x + 3 * y']
         seeds = [1, 2]
         z = Reverse(var, fcts, seeds)
         -- Values -- 
         Function F1: 9.38905609893065
         Function F2: 7.0
         -- Gradients -- 
         Function F1: [16.7781122  8.3890561]
         Function F2: [2. 6.]
         """
        check = [1 if isinstance(i, str) else 0 for i in fct]
        if len(check) != sum(check):
            raise TypeError('Each function should be a string!')
        else:
            if isinstance(var, dict):
                if seed is None:
                    # default seed
                    seed = [1] * len(fct)
                
                num_var = len(var)
                self.var2idx = dict(zip(list(var.keys()), list(range(num_var))))
                                
                # static methods
                static_methods = ['log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']

                num_fct = len(fct)
                self.der = []
                self.var = []
                for idx, func in enumerate(fct):
                    # reset variables for each functions
                    for var_name, var_value in var.items():
                        exec(f'{var_name} = Reverse_Mode(float(var_value))')

                    # update functions naming

                    for i in static_methods:
                        if i in func:
                            # for multiple but same static methods 
                            func = re.sub(i + r'\(', 'Reverse_Mode.' + i + '(', func)
                            func = re.sub('arcReverse_Mode.', 'arc', func)

                    f1 = eval(func)
                    out1, out2 = eval(f'f1.derivative(' + str(list(var.keys())).replace('\'','') + ', seed[idx])')
                    # update the two attributes
                    self.var.append(out1)
                    self.der.append(out2)
            
                self.der = np.array(self.der)
                self.var = np.array(self.var)

            else: 
                raise TypeError('The variable should be a dictionary!')

    def __repr__(self):
        output_string = '-- Values -- \n'
        for var_idx in range(self.var.shape[0]):
            output_string =  output_string + 'Function F' + str(var_idx + 1) + ': ' + str(self.var[var_idx]) + '\n'

        output_string = output_string + '-- Gradients -- \n'

        for fct_idx in range(self.der.shape[0]):
            output_string = output_string + 'Function F' + str(fct_idx + 1) + ': ' + str(self.der[fct_idx]) + '\n'

        return output_string

    def __str__(self):
        output_string = '-- Values -- \n'
        for var_idx in range(self.var.shape[0]):
            output_string =  output_string + 'Function F' + str(var_idx + 1) + ': ' + str(self.var[var_idx]) + '\n'

        output_string = output_string + '-- Gradients -- \n'

        for fct_idx in range(self.der.shape[0]):
            output_string = output_string + 'Function F' + str(fct_idx + 1) + ': ' + str(self.der[fct_idx]) + '\n'

        return output_string

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

        return value, derivatives

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
