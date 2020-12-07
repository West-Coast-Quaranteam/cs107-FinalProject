import numpy as np
import string
from apollo_ad.apollo_ad import *

def apollo_ad_UI():
    print('Welcome to Apollod AD Library!')
    print('Enter the number of variables:')
    num_var = input()
    try:
        num_var = int(num_var)
    except:
        raise AttributeError("Please only type in integers!")

    id2var = list(string.ascii_lowercase)
    for i in range(num_var):
        print('Type the value of variable No. ' + str(i+1) +': (It must be a float)')
        val = input()
        print('Type the derivative seed of variable No. ' + str(i+1) +'. It must be a float: ')
        der = input()
        der_ = np.zeros((num_var,))
        der_[i] = float(der)
        exec(f'{id2var[i]} = Variable(float(val), der_)')

    print('Enter the number of functions:')
    num_fct = input()
    try:
        num_fct = int(num_fct)
    except:
        raise AttributeError("Please only type in integers!")

    fcts = []
    
    print("-- Important Note: the name of the variable in the function should follow alphabetical order. E.g. 1st variable: a; 2nd: b, ... 26th variable: z. While the UI only supports up to 26 variables, for more variables, directly use the function class. --")
    static_methods = ['log', 'sqrt', 'exp', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh', 'tanh']
    for i in range(num_fct):
        print('Type function No. ' + str(i+1) +' :')
        fct = input()
        for i in static_methods:
            if i in fct:
                fct = fct[:fct.find(i)] + 'Variable.' + fct[fct.find(i):]
        fcts.append(eval(fct))

    f = Functions(fcts)
    print(f)
    return f

apollo_ad_UI()