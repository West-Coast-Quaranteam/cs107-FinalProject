# cs107-FinalProject, Group 16

[![Build Status](https://travis-ci.com/West-Coast-Quaranteam/cs107-FinalProject.svg?token=z1QwjsA3zqLzUQzz5VsE&branch=master)](https://travis-ci.com/West-Coast-Quaranteam/cs107-FinalProject)
[![codecov](https://codecov.io/gh/West-Coast-Quaranteam/cs107-FinalProject/branch/master/graph/badge.svg?token=NY1T0T5UG3)](undefined)

| Group members   | Email |          
| ----------------|:-----:| 
| Connor Capitolo | connorcapitolo@g.harvard.edu |
| Kexin Huang     | kexinhuang@hsph.harvard.edu  |
| Haoxin Li       | haoxin_li@hsph.harvard.edu   | 
| Chen Lucy Zhang | chz522@g.harvard.edu         | 


## Broader Impact

This python apollo_ad package is user-friendly, allowing users to use the package as a convenient problem-solving tool in automatically calculating derivatives in multiple ways. With this package, users can reach the goal without coding every step, especially when they are dealing with large-scale computation and complex cases. Even users with little experience could operate this package by simply input values in terminal UI. Variables and derivatives input allows the functions of autodifferency to differentiate both complex functions and basic functions. At the same time, The range of this package is wide since it allows users to create user-specific functions on their own. The package is impactful due to its ability to easily calculate user-specific functions inheriting instances of variables using the apollo_ad package.

The package would potentially be used within machine learning projects, especially for users making use of automatic differentiation, since the package will also decide whether it is more efficient to use reverse mode or forward mode based on the user’s input. This further extension also creates deeper implications in both scientific research and real-word applications.

For education purposes, this package could also be used in higher education or middle school, such as the students who are studying calculus and derivatives. This package could be used like a validation package to see students’ accuracy. The package will tell accurate answers without step-by-step calculation, which is fast and efficient. On the other side, the  package could be harmful for a beginner since they may use it like a shortcut to get answers without diving deep to explore the methods.

For complex ethical consideration, every method has its two sides. The package is impactful due to its ability to easily calculate user-specific functions. There are also possibilities that user may cause the negative effects, since no information is given about the exact purposes, whether it is legal or not. However, since the impact is wide, there may be ethical concerns which are far from original expectations.

## Inclusivity Statement

Our project mission is to make it universally accessible and useful, as a simple math tool for everyone. We are committed to creating a diverse, equal and inclusive workforce. Our software is open source on Github and PyPI for everyone, enabling creating pull requests and adding comments. 

Further, we endeavor to build packages that work for everyone by including perspectives from backgrounds that vary by race, ethnicity, religion, gender, age, disability, social backgrounds, sexual orientation, culture, and national origin. Users with any identity could use our packages as the way they want, without releasing their personal information. With possible user adds on, we encourage further contribution on multi-language versions, empowering a diverse participation.


## Installation (we assume you are familiar with virtual environments and Git)

```bash
# install virtualenv
pip install virtualenv

# create virtual environment
virtualenv apollo_ad_env

# activate virtual environment
source apollo_ad_env/bin/activate

# Method 1 Github:
# clone from GitHub
git clone https://github.com/West-Coast-Quaranteam/cs107-FinalProject.git

# get into the folder
cd cs107-FinalProject

# install requirements
pip install -r requirements.txt

# Test the package
# From directory apollo_ad/tests/ run the module test.py
pytest test_scalar.py

# Method 2 PyPI:
# Go to the PyPI link:  https://pypi.org/project/apollo-ad/0.0.3/
download, install and test:
pip install apollo-ad==0.0.3
pytest test_scalar.py
```

## Examples

We show two examples here to use `apollo_ad`. 

First, to calculate the derivative of y = cos(x) + x^2 at x = 2:

```python
from apollo_ad.apollo_ad import *
x = Variable(2)
y = Variable.cos(x) + x ** 2

print(y) # Value: 3.5838531634528574 , Der: [3.09070257]
print(y.var) # 3.583853163452857
print(y.der) # [3.09070257]

assert y.var == np.cos(2) + 4
assert y.der == -np.sin(2) + 2 * 2
```

Second, to calculate the derivative of y = 2 * log(x) - sqrt(x) / 3 at x = 2:

```python
from apollo_ad.apollo_ad import *
x = Variable(2) 
y = 2 * Variable.log(x) - Variable.sqrt(x)/3

print(y) # Value: 0.9148898403288588 , Der: [0.88214887]
print(y.var) # 0.9148898403288588
print(y.der) # [0.88214887]

assert np.around(y.var, 4) == np.around(2 * np.log(2) - np.sqrt(2)/3, 4)
assert np.around(y.der, 4) == np.around(2 * 1/2 - 1/3 * 1/2 * 2**(-1/2), 4) 
```

You can also run the above examples by:
```python
python demo.py
```
