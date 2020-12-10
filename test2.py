from apollo_ad.apollo_ad import *

## Forward Mode

var = {'x': 3, 'y': 4}
fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3 + log(x)']
z = Forward(var, fcts)

print(z)


var = {'x': 3, 'y': 4}
seeds = {'x': 1, 'y': 2}
fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
z = Forward(var, fcts, seeds)

print(z)


var = {'x': 1, 'y': 2}
fcts = ['x * y + exp(x * y)', 'x + 3 * y']
z = Reverse(var, fcts)

print(z)

var = {'x': 1, 'y': 2}
fcts = ['x * y + exp(x * y)', 'x + 3 * y']
seeds = [1, 2]
z = Reverse(var, fcts, seeds)

print(z)

vars = {'x': 0.5, 'y': 4}
fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3', 'sqrt(x)/3', '3 * sinh(x) - 4 * arcsin(x) + 5']
z = auto_diff(vars, fcts)

print(z)

vars = {'x': 3, 'y': 4, 'z': 5}
fcts = ['cos(x) + y ** 2 + sqrt(z)']
z = auto_diff(vars, fcts)

print(z)

vars = {'x': 3, 'y': 4}
seeds = {'x': 1, 'y': 2}
fcts = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
z = auto_diff(vars, fcts, seeds)

print(z)