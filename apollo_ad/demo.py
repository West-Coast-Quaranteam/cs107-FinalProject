from .apollo_ad import *

def demo():
	print('--- Example auto_diff ---')

	var = {'x': 0.5, 'y': 4}
	fct = ['cos(x) + y ** 2', 
			'2 * log(y) - sqrt(x)/3', 
			'sqrt(x)/3', 
			'3 * sinh(x) - 4 * arcsin(x) + 5']

	out = auto_diff(var, fct)
	print(out)

	print('--- Example Forward ---')

	var = {'x': 3, 'y': 4}
	fct = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
	out = Forward(var, fct)
	print(out)

	print('--- Example Reverse ---')

	var = {'x': 1, 'y': 2}
	fct = ['x * y + exp(x * y)', 'x + 3 * y']
	out = Reverse(var, fct)
	print(out)

	print('--- Example Seed ---')

	var = {'x': 3, 'y': 4}
	seed = {'x': 1, 'y': 2}
	fct = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
	z = auto_diff(var, fct, seed)
	print(z)

