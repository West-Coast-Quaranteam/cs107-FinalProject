import numpy as np

class Reverse_Mode:
    def __init__(self, var, seed = 1):
        self.var = var
        # need this because we need to store a form of the computational graph
        self.child = []
        self.der = seed

    def grad(self):
        # base case for recursion: basically, we know that if node i has no child, then it's the output variable
        if len(self.child) != 0:
            # I'm using this based on the equation for lecture 12 we had in class
            df_dui = 0
            for duj_dui, df_duj in self.child:
                df_dui += duj_dui * df_duj.grad()
            self.der = df_dui
        return self.der

    def __add__(self, other):
        try:
            f = Reverse_Mode(self.var + other.var)
            other.child.append((1.0, f))
            self.child.append((1.0, f))
        except AttributeError:
            f = Reverse_Mode(self.var + other)
        return f

    def __radd__(self, other):
        return self.__add__(other)

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
    # def __sub__(self, other):
    #     return self.__add__(-other)

    # def __rsub__(self, other):
    #     return (-self).__add__(other)

    # def __neg__(self):
    #     # children = self.child
    #     # f = Reverse_Mode(-self.var, -self.der)
    #     # f.child = children
    #     self.der = -self.der
    #     self.var = -self.var
    #     return self


    def sin(self):
        f = Reverse_Mode(np.sin(self.var))
        self.child.append((np.cos(self.var), f))
        return f

    def cos(self):
        f = Reverse_Mode(np.cos(self.var))
        self.child.append(-np.sin(self.var))
        return f

    
# HAVING ISSUES WITH NEGATION/SUBTRACTION
# x = Reverse_Mode(2)
# y = Reverse_Mode(3)

# b = x-4
# f = b-y
# print(f.var)
# print(y.grad())
