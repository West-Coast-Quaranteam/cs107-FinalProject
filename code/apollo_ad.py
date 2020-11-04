class Variable:
    def __init__(self, var, seed=1):
        self.var = var
        self.der = seed

    def __add__(self, other):
        try:
            return Variable(self.var + other.var, self.der + other.der)
        except AttributeError:
            return Variable(self.var + other, self.der)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        try:
            return Variable(self.var * other.var, self.var * other.der + self.der * other.var)
        except AttributeError:
            return Variable(self.var * other, self.der * other)

    def __rmul__(self, other):
        return self.__mul__(other)


if __name__ == "__main__":
    a = 2
    alpha = 2.0
    beta = 3.0

    x = Variable(a)
    f = alpha * x + beta
    print(f.var, f.der)
    assert f.var == 7.0 and f.der == 2.0

    f = x * alpha + beta
    assert f.var == 7.0 and f.der == 2.0

    f = beta + alpha * x
    assert f.var == 7.0 and f.der == 2.0

    f = beta + x * alpha
    assert f.var == 7.0 and f.der == 2.0