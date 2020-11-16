# Test suite 
# import sys
# sys.path.insert(
#     1, '/Users/kirklandtomato/Classes/CS107-FinalProject/code/test')
# sys.path.append('../')

import numpy as np

import apollo_ad as ap


def test_apollo_ad_functions():

	def test_sin():
		# Expect value of 18.42, derivative of 6.0
		x = ap.Variable(np.pi / 2)
		f = 3 * 2 * np.sin(x) + 6 * x + 3

		assert np.round(f.val, 2) == [18.42]
		assert np.round(f.der, 2) == [6.0]


		x2 = ap.Variable(np.pi)
		f2 = 3 * np.sin(x2) +  3
		assert np.round(f2.val, 2) == [3.0]
		assert f2.der == [-3.0]


	def test_cos():
		# Expect value of -10pi, derivative of 0.0 (because of -sin(pi))
		x = ap.Variable(np.pi)
		f = 5 * np.cos(x)+ np.cos(x) * 5
		assert f.val == [-10]
		assert abs(f.der) <= 1e-14

	def test_arcsin():
		x = ap.Variable(0)
		f = np.arcsin(x)
		assert f.val == [0.0]
		assert f.der == [1.0]

		# Domain of arcsin(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = ap.Variable(-1.01)
			np.arcsin(x)

	def test_arccos():
		x = ap.Variable(0)
		f = np.arccos(x)
		assert f.val == [np.pi / 2]
		assert f.der == [-1.0]

		# Domain of arccos(x) is -1 <= x <= 1
		with np.testing.assert_raises(ValueError):
			x = ap.Variable(1.01)
			np.arccos(x)