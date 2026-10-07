#!/usr/bin/env python
# coding: utf-8
'''


@author: Stefan Wallner
'''

from __future__ import absolute_import, print_function, division

import os
import unittest

import sphysics

import numpy as np
import tensorflow as tf



class LorentzTest(unittest.TestCase):

	def __init__(self, methodName: str = ...) -> None:
		super().__init__(methodName)
		data = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.npz'))
		self._p1_np = data['p1']
		self._p1_tf = tf.constant(self._p1_np)


	def test_lp_np(self):
		'''
		Test lorentz product of numpy arrays
		'''
		s = sphysics.lorentz.lp(self._p1_np, self._p1_np)

		self.assertIsInstance(s, np.ndarray)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-sphysics.Constants.M.pi**2)), 0., 6)


	def test_lp_tf(self):
		'''
		Test lorentz product of tensorflow arrays
		'''
		s = sphysics.lorentz.lp(self._p1_tf, self._p1_tf)

		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-sphysics.Constants.M.pi**2)).numpy(), 0., 6)


	def test_tensorproduct_np(self):
		'''
		Test outer tensor product
		'''
		p1p1 = sphysics.lorentz.tensorproduct(self._p1_np, self._p1_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(p1p1[:,3,:]/self._p1_np[3] - self._p1_np)), 0., 6)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(p1p1[2,:,:]/self._p1_np[2] - self._p1_np)), 0., 6)


	def test_tensorproduct_tf(self):
		'''
		Test outer tensor product
		'''
		p1p1 = sphysics.lorentz.tensorproduct(self._p1_tf, self._p1_tf)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(p1p1[:,3,:]/self._p1_tf[3] - self._p1_tf)).numpy(), 0., 6)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(p1p1[2,:,:]/self._p1_tf[2] - self._p1_tf)).numpy(), 0., 6)


	def test_epsilon(self):
		"""
		Checks antisymmetry of epsilon and a few other properties
		"""
		epsilon = sphysics.lorentz.epsilon
		et = sphysics.lorentz.et
		for i in range(4):
			for j in range(4):
				for k in range(4):
					for l in range(4):
						self.assertEqual(epsilon[i, j, k, l], -epsilon[j, i, k, l], f"ERROR in {i}, {j}, {k}, {l}")
						self.assertEqual(epsilon[i, j, k, l], -epsilon[l, j, k, i], f"ERROR in {i}, {j}, {k}, {l}")
						self.assertEqual(epsilon[i, j, k, l], -epsilon[i, k, j, l], f"ERROR in {i}, {j}, {k}, {l}")
						self.assertEqual(epsilon[i, j, k, l], -epsilon[i, l, k, j], f"ERROR in {i}, {j}, {k}, {l}")
						self.assertEqual(epsilon[i, j, k, l], -epsilon[i, j, l, k], f"ERROR in {i}, {j}, {k}, {l}")

		contraction = np.einsum('abgd,abgd,a,b,g,d->',
								epsilon, epsilon, et, et, et, et)
		self.assertEqual(contraction, -24, "Contraction of epsilon is {0} instead of -24")

		contraction = np.einsum('abgd,rbgd,a,b,g,d -> ar',
								epsilon, epsilon, et, et, et, et)
		self.assertTrue(np.all(contraction == np.identity(4)*(-6)), "Contraction 'abgd,rbgd -> ar' is\n{0}\ninstread of diag(-6).".format(contraction))


	def test_boost(self):
		np.seterr(all='raise')
		np.random.seed(83838)
		m1 = 0.139
		m2 = 0.5
		v1_np = np.random.uniform(1, 2, size=(3,100_000))
		v2_np = np.random.uniform(1, 2, size=(3,100_000))
		p1_np = np.empty((4,v1_np.shape[1]))
		p1_np[1:] = v1_np
		p1_np[0] = sphysics.math.sqrt(m1**2 + sphysics.math.sum(v1_np**2, axis=0))
		p2_np = np.empty((4,v2_np.shape[1]))
		p2_np[1:] = v2_np
		p2_np[0] = sphysics.math.sqrt(m2**2 + sphysics.math.sum(v2_np**2, axis=0))

		ptot = p1_np+p2_np
		boost = sphysics.lorentz.getBoostToRestFrame(ptot)
		p1_boosted = sphysics.lorentz.applyBoost(boost, p1_np)
		p2_boosted = sphysics.lorentz.applyBoost(boost, p2_np)
		inv_mass_squared = sphysics.lorentz.lp(ptot,ptot)
		inv_mass_squared_boosted = sphysics.lorentz.lp(p1_boosted+p2_boosted,p1_boosted+p2_boosted)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs((p1_boosted+p2_boosted)[1])), 0., 6)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs((p1_boosted+p2_boosted)[2])), 0., 6)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs((p1_boosted+p2_boosted)[3])), 0., 6)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(inv_mass_squared-inv_mass_squared_boosted)), 0., 6)



if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
