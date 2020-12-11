import numpy as np
import string
from .apollo_ad import *

def UI():
    '''
    Welcome to Apollod AD Library!
    Enter the number of variables:
    2
    Enter the number of functions:
    3
    Type the variable name of variable No. 1: 
    a
    Type the value of variable a: (It must be a float)
    3
    Type the derivative seed of variable a. It must be a float: 
    1
    Type the variable name of variable No. 2: 
    b
    Type the value of variable b: (It must be a float)
    2
    Type the derivative seed of variable b. It must be a float: 
    1
    Type function No. 1 :
    a + b + sin(b)
    Type function No. 2 :
    sqrt(a) + log(b)
    Type function No. 3 :
    exp(a * b) + a ** 2
    ---- Summary ----
    Variable(s):
    {'a': '3', 'b': '2'}
    Function(s): 
    a + b + sin(b)
    sqrt(a) + log(b)
    exp(a * b) + a ** 2
    ---- Computing Gradients ----
    # of variables < # of functions ====> automatically use the forward mode!
    ---- Output ----
    -- Values -- 
    Function F1: 5.909297426825682
    Function F2: 2.4251979881288226
    Function F3: 412.4287934927351
    -- Gradients -- 
    Function F1: [1.         0.58385316]
    Function F2: [0.28867513 0.5       ]
    Function F3: [ 812.85758699 1210.28638048]

    '''
    print('Welcome to Apollod AD Library!')
    print('Enter the number of variables:')
    num_var = input()
    try:
        num_var = int(num_var)
    except:
        raise AttributeError("Please only type in integers!")

    print('Enter the number of functions:')
    num_fct = input()
    try:
        num_fct = int(num_fct)
    except:
        raise AttributeError("Please only type in integers!")

    if num_var <= num_fct:
        forward_mode = True
    else:
        forward_mode = False

    variable_input = {}
    if forward_mode:
        seeds = {}
    else:
        seeds = []

    for i in range(num_var):
        print('Type the variable name of variable No. '  + str(i+1) + ': ')
        name = input()
        print('Type the value of variable '+ name +': (It must be a float)')
        val = input()

        variable_input[name] = val

        if forward_mode:
            print('Type the derivative seed of variable '+ name +'. It must be a float: ')
            der = input()
            seeds[name] = float(der)

    fcts = []
    for i in range(num_fct):
        print('Type function No. ' + str(i+1) +' :')
        fct = input()
        fcts.append(fct)

        if not forward_mode:
            print('Type the seed of function '+ str(i+1) +'. It must be a float: ')
            der = input()
            seeds.append(float(der))

    print('---- Summary ----')
    print('Variable(s):')
    print(variable_input)
    print('Function(s): ')
    print('\n'.join(fcts))
    print('Seeds(s): ')
    print(seeds)
    print('---- Computing Gradients ----')
    f = auto_diff(variable_input, fcts, seeds)
    print('---- Output ----')
    print(f)
    return f