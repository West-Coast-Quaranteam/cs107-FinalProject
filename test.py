from apollo_ad.apollo_ad import *

x = Variable(3, [1, 0])
y = Variable(4, [0, 1])
f1 =  Variable.cos(x) + y ** 2

print(f1) # Value: 15.010007503399555 , Der: [-0.14112001  8.        ]

f2 = f1 + Variable.cos(x)
print(f2) # Value: 14.02001500679911 , Der: [-0.28224002  8.        ]

x = Variable(3, [1, 0])
y = Variable(4, [0, 1])
f1 =  Variable.cos(x) + y ** 2
f2 = 2 * Variable.log(y) - Variable.sqrt(x)/3 
z = Functions([f1, f2])

print(z)
# Value: [15.0100075   2.19523845] , 
#	Der: [[-0.14112001  8.        ]
#         [-0.09622504  0.5       ]]


x = Variable(3, [1, 0, 0])
y = Variable(4, [0, 1, 0])
z = Variable(5, [0, 0, 1])
f1 =  Variable.cos(x) + y ** 2 + Variable.sin(z)
f2 = 2 * Variable.log(y) - Variable.sqrt(x)/3 
z = Functions([f1, f2])

print(z)
# Value: [14.05108323  2.19523845] , 
# Der: 
# [[-0.14112001  8.          0.28366219]
#  [-0.09622504  0.5         0.        ]]

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

## Reverse Mode

var = {'x': 1, 'y': 2}
fcts = ['x * y + exp(x * y)', 'x + 3 * y']
z = Reverse(var, fcts)

print(z)

var = {'x': 1, 'y': 2}
fcts = ['x * y + exp(x * y)', 'x + 3 * y']
seeds = [1, 2]
z = Reverse(var, fcts, seeds)

print(z)

## Automatic Identification

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