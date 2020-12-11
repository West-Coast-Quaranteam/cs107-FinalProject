# cs107-FinalProject, Group 16

[![Build Status](https://travis-ci.com/West-Coast-Quaranteam/cs107-FinalProject.svg?token=z1QwjsA3zqLzUQzz5VsE&branch=master)](https://travis-ci.com/West-Coast-Quaranteam/cs107-FinalProject) 
[![codecov](https://codecov.io/gh/West-Coast-Quaranteam/cs107-FinalProject/branch/master/graph/badge.svg?token=NY1T0T5UG3)](undefined)

| Group members   | Email |          
| ----------------|:-----:| 
| Connor Capitolo | connorcapitolo@g.harvard.edu |
| Kexin Huang     | kexinhuang@hsph.harvard.edu  |
| Haoxin Li       | haoxin_li@hsph.harvard.edu   | 
| Chen Lucy Zhang | chz522@g.harvard.edu         | 


## Installation (we assume you are familiar with virtual environments and Git)

To install from pip:

```bash
pip install apollo_ad
```

To install from source:

```bash
# install virtualenv
pip install virtualenv

# create virtual environment
virtualenv apollo_ad_env

# activate virtual environment
source apollo_ad_env/bin/activate

# clone from GitHub
git clone https://github.com/West-Coast-Quaranteam/cs107-FinalProject.git

# get into the folder
cd cs107-FinalProject

# install requirements
pip install -r requirements.txt
```

## Running Python Interactive Session From Command Line (we assume you have Python properly installed)
From your command line, please run:
```bash
python
```

This is the prompt you should now see:
```bash
>>>
```

You're ready to use Apollo AD!

## Examples

#### UI

We also provide a nice interface, where you can specify the variable values and functions without any coding:

```python
from apollo_ad import UI
UI()
```

<details>
  <summary>Click here for an UI interaction example!</summary>


```
Welcome to Apollod AD Library!
Enter the number of variables:
2
Enter the number of functions:
3
Type the variable name of variable No. 1: 
a
Type the value of variable a (Please only input a float):
3
Type the derivative seed of variable a (Please only input a float; your default input should be 1): 
1
Type the variable name of variable No. 2: 
b
Type the value of variable b (Please only input a float):
2
Type the derivative seed of variable b (Please only input a float; your default input should be 1): 
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
```

</details>

#### Programming Usage

`apollo_ad` expects two inputs, a dictionary with variable name as the key and variable value as the value and a list of strings where each string describes a function:


```python
from apollo_ad import auto_diff
var = {'x': 0.5, 'y': 4}
fct = ['cos(x) + y ** 2', 
		'2 * log(y) - sqrt(x)/3', 
		'sqrt(x)/3', 
		'3 * sinh(x) - 4 * arcsin(x) + 5']

out = auto_diff(var, fct)
print(out)

# -- Values -- 
# Function F1: 16.87758256189037
# Function F2: 2.5368864618442655
# Function F3: 0.23570226039551587
# Function F4: 4.468890814088047
# -- Gradients -- 
# Function F1: [-0.47942554  8.        ]
# Function F2: [-0.23570226  0.5       ]
# Function F3: [0.23570226 0.        ]
# Function F4: [-1.23592426 -4.61880215]
```

`apollo_ad` supports both forward and reverse mode in the backend, where the `auto_diff` class autoamtically detects which is the best way to use, depending on the number of inputs and outputs. You can also directly use the `Forward` and `Reverse` mode class. 

```python
from apollo_ad import Forward, Reverse
var = {'x': 3, 'y': 4}
fct = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
out = Forward(var, fct)
print(out)

# -- Values -- 
# Function F1: 15.010007503399555
# Function F2: 3.2938507417182654
# -- Gradients -- 
# Function F1: [-0.14112001  8.        ]
# Function F2: [0.23710829 0.5       ]

var = {'x': 1, 'y': 2}
fct = ['x * y + exp(x * y)', 'x + 3 * y']
out = Reverse(var, fct)
print(out)

# -- Values -- 
# Function F1: 9.38905609893065
# Function F2: 7.0
# -- Gradients -- 
# Function F1: [16.7781122  8.3890561]
# Function F2: [1. 3.]
```

You can also specify the seed. As the seed in forward mode is for each variable whereas in reverse mode it's for each function, we expect a dictionary for forward mode and a list for reverse mode. Here is an example:

```python
var = {'x': 3, 'y': 4}
seed = {'x': 1, 'y': 2}
fct = ['cos(x) + y ** 2', '2 * log(y) - sqrt(x)/3']
z = auto_diff(var, fct, seed)
# -- Values -- 
# Function F1: 15.010007503399555
# Function F2: 2.1952384530501554
# -- Gradients -- 
# Function F1: [-0.14112001 16.        ]
# Function F2: [-0.09622504  1.        ]
```

You can also run the above examples by typing:

```python
from apollo_ad import demo
demo()
```

## Additional Notes
If you are trying to access the attributes of this class, please note that they will be lists, numpy arrays, or dictionaries.

Most importantly, have fun and provide us feedback!