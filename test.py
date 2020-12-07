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