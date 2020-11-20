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

# Test the package
# Run module test.py in apollo_ad/tests/ directory
pytest test.py
```

## Examples

We show two examples here to use `apollo_ad`. 

First, to calculate the derivative of y = cos(x) + x^2 at x = 2:

```python
from apollo_ad.apollo_ad import *
x = Variable(2)
y = Variable.cos(x) + x ** 2
print(y)
print(y.var)
print(y.der)
# y.var: 3.583853163452857 y.der: [3.09070257]
assert y.var == np.cos(2) + 4
assert y.der == -np.sin(2) + 2 * 2
```

Second, to calculate the derivative of y = 2 * log(x) - sqrt(x) / 3 at x = 2:

```python
from apollo_ad.apollo_ad import *
x = Variable(2) 
y = 2 * Variable.log(x) - Variable.sqrt(x)/3

print(y)
print(y.var)
print(y.der)

# Value: 0.9148898403288588 , Der: [0.88214887]
assert np.around(y.var, 4) == np.around(2 * np.log(2) - np.sqrt(2)/3, 4)
assert np.around(y.der, 4) == np.around(2 * 1/2 - 1/3 * 1/2 * 2**(-1/2), 4) 
```