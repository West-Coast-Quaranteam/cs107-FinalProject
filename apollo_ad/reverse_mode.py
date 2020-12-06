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

    def sin(self):
        f = Reverse_Mode(np.sin(x.var))
        self.child.append((np.cos(x.var), f))
        return f


x = Reverse_Mode(0.5)
y = Reverse_Mode(4.2)
# does multiplication, then sin, then addition
# f = x * y + Reverse_Mode.sin(x)

# assert abs(f.var - 2.579425538604203) <= 1e-15
# assert abs(x.grad() - (y.var + np.cos(x.var))) <= 1e-15
# assert abs(y.grad() - x.var) <= 1e-15

# f = 2 * x * y + Reverse_Mode.sin(x)

# assert abs(f.var - (2 * 0.5 * 4.2 + np.sin(0.5))) <= 1e-15
# assert abs(x.grad() - (2 * y.var + np.cos(x.var))) <= 1e-15
# assert abs(y.grad() - 2 * x.var) <= 1e-15

# f = 2 * x * y + 3 + Reverse_Mode.sin(x)

# assert abs(f.var - (2 * 0.5 * 4.2 + np.sin(0.5) + 3)) <= 1e-15
# assert abs(x.grad() - (2 * y.var + np.cos(x.var))) <= 1e-15
# assert abs(y.grad() - 2 * x.var) <= 1e-15

f = 4*x+Reverse_Mode.sin(x)+6*y

assert np.round(f.var,4) == 27.6794 and np.round(x.grad(), 4) == 4.8776 and np.round(y.grad(), 4) == 6.0000













'''
import numpy as np

class Reverse_Mode:
    # how should the seed be set for this?
    def __init__(self, var, seed = 1):
        self.var = var
        self.child = []
        self.der = seed

    # this gets called on the way back
    # how does it handle child?
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
        except AttributeError:
            f = Reverse_Mode(self.var + other)
        self.child.append((1.0, f))
        return f

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            f = Reverse_Mode(self.var * other.var)
            other.child.append((self.var, f)) # in this case, appending this to y
        except AttributeError:
            f = Reverse_Mode(self.var * other)
        self.child.append((other, f))
        return f

    def __rmul__(self, other):
        return self.__mul__(other)


    def sin(self):
        f = Reverse_Mode(np.sin(x.var))
        self.child.append((np.cos(x.var), f))
        return f


x = Reverse_Mode(0.5)
y = Reverse_Mode(4.2)
# does multiplication, then sin, then addition
# f = x * y + Reverse_Mode.sin(x)
# assert abs(f.var - 2.579425538604203) <= 1e-15
# assert abs(x.grad() - (y.var + np.cos(x.var))) <= 1e-15
# assert abs(y.grad() - x.var) <= 1e-15

# f = 2 * x
# print(x.grad()) #should return 2

f = x * y + Reverse_Mode.sin(x)
print(f.var)
print(x.grad())
print(y.grad())
#assert abs(y.grad() - 2 * x.var) <= 1e-15
'''
