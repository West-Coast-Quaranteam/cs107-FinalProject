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
        return value, derivatives

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
            return Variable(self.var / other.var, (self.der * other.var - self.var * other.der) / (other.var ** 2))
        except AttributeError:
            return Variable(self.var / other, self.der / other)

    def sin(self):
        f = Reverse_Mode(np.sin(self.var))
        self.child.append((np.cos(self.var), f))
        return f

    def cos(self):
        f = Reverse_Mode(np.cos(self.var))
        self.child.append((-np.sin(self.var), f))
        return f

    def exp(self):
        f = Reverse_Mode(np.exp(self.var))
        self.child.append((np.exp(self.var), f))
        return f

    def tanh(self):
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
        f = Reverse_Mode(np.tanh(self.var))
        tanh_der = 1 / np.power(np.cosh(self.var), 2)
        self.child.append((tanh_der, f))

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

if __name__ == '__main__':
    x = Reverse_Mode(1)
    inputs = [x]
    f = x - 4

    value, check = f.derivative(inputs)
    assert np.round(value, 4) == -3.0000
    assert np.round(check[0], 4) == 1.0000

    # x = Reverse_Mode(1)
    # f = x
    # inputs = [x]
    # value, check = f.derivative(inputs)
    # print(value, check)
    # assert np.round(value, 4) == 13.0000
    # assert np.round(check[0], 4) == 1.0000
    # assert np.round(check[1], 4) == 3.0000
    # assert np.round(check[2], 4) == 2.0000
