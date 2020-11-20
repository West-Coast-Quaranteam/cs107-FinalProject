from apollo_ad.apollo_ad import *

print('--- Example 1 ---')

x = Variable(2)
y = Variable.cos(x) + x ** 2

print(y) # Value: 3.5838531634528574 , Der: [3.09070257]
print(y.var) # 3.583853163452857
print(y.der) # [3.09070257]

assert y.var == np.cos(2) + 4
assert y.der == -np.sin(2) + 2 * 2

print('--- Example 2 ---')

x = Variable(2) 
y = 2 * Variable.log(x) - Variable.sqrt(x)/3

print(y) # Value: 0.9148898403288588 , Der: [0.88214887]
print(y.var) # 0.9148898403288588
print(y.der) # [0.88214887]

assert np.around(y.var, 4) == np.around(2 * np.log(2) - np.sqrt(2)/3, 4)
assert np.around(y.der, 4) == np.around(2 * 1/2 - 1/3 * 1/2 * 2**(-1/2), 4) 