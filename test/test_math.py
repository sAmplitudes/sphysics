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


class MathTest(unittest.TestCase):
	def __init__(self, methodName: str = ...) -> None:
		super().__init__(methodName)
		self._array_np = np.linspace(0., 5., 100)
		self._array_tf = tf.constant(self._array_np)


	def test_sqrt(self):
		s = sphysics.math.sqrt(self._array_np)
		self.assertIsInstance(s, np.ndarray)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.sqrt(self._array_np))), 0., 6)

		s = sphysics.math.sqrt(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.sqrt(self._array_np))).numpy(), 0., 6)

	def test_divide(self):
		s = sphysics.math.divide(self._array_np, 1.+self._array_np)
		self.assertIsInstance(s, np.ndarray)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-self._array_np/(1.+self._array_np))), 0., 6)

		s = sphysics.math.divide(self._array_tf, 1.+self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-self._array_np/(1.+self._array_np))).numpy(), 0., 6)

		s = sphysics.math.divide(1.0+self._array_np, self._array_np, where=(self._array_np!=0.), whereNotValue=20.)
		self.assertAlmostEqual(s[0], 20., 6)

		s = sphysics.math.divide(1.0+self._array_tf, self._array_tf, where=(self._array_tf!=0.), whereNotValue=20.)
		self.assertAlmostEqual(s[0].numpy(), 20., 6)

	def test_min(self):
		s = sphysics.math.min(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.min(self._array_np))), 0., 6)

		s = sphysics.math.min(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.min(self._array_np))).numpy(), 0., 6)


	def test_max(self):
		s = sphysics.math.max(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.max(self._array_np))), 0., 6)

		s = sphysics.math.max(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.max(self._array_np))).numpy(), 0., 6)


	def test_abs(self):
		s = sphysics.math.abs(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.abs(self._array_np))), 0., 6)

		s = sphysics.math.abs(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.abs(self._array_np))).numpy(), 0., 6)

	def test_abs2(self):
		s = sphysics.math.abs2(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.abs(self._array_np)**2)), 0., 6)

		s = sphysics.math.abs2(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.abs(self._array_np)**2)).numpy(), 0., 6)


	def test_mean(self):
		s = sphysics.math.mean(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.mean(self._array_np))), 0., 6)

		s = sphysics.math.mean(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.mean(self._array_np))).numpy(), 0., 6)


	def test_cos(self):
		s = sphysics.math.cos(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.cos(self._array_np))), 0., 6)

		s = sphysics.math.cos(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.cos(self._array_np))).numpy(), 0., 6)


	def test_sin(self):
		s = sphysics.math.sin(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.sin(self._array_np))), 0., 6)

		s = sphysics.math.sin(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.sin(self._array_np))).numpy(), 0., 6)


	def test_arctan(self):
		s = sphysics.math.arctan(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.arctan(self._array_np))), 0., 6)

		s = sphysics.math.arctan(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.arctan(self._array_np))).numpy(), 0., 6)


	def test_arctan2(self):
		s = sphysics.math.arctan2(self._array_np, self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.arctan2(self._array_np, self._array_np))), 0., 6)

		s = sphysics.math.arctan2(self._array_tf, self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.arctan2(self._array_np, self._array_np))).numpy(), 0., 6)


	def test_exp(self):
		s = sphysics.math.exp(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.exp(self._array_np))), 0., 6)

		s = sphysics.math.exp(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.exp(self._array_np))).numpy(), 0., 6)

	def test_log(self):
		s = sphysics.math.log(self._array_np[self._array_np>0])
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.log(self._array_np[self._array_np>0]))), 0., 6)

		s = sphysics.math.log(self._array_tf[self._array_np>0])
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.log(self._array_np[self._array_np>0]))).numpy(), 0., 6)

	def test_pow(self):
		s = sphysics.math.pow(self._array_np, 4.)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.power(self._array_np, 4.))), 0., 6)

		s = sphysics.math.pow(self._array_tf, 4.)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.power(self._array_np, 4.))).numpy(), 0., 6)


	def test_sum(self):
		s = sphysics.math.sum(self._array_np)
		self.assertAlmostEqual(sphysics.math.abs(s-np.sum(self._array_np)), 0., 6)

		s = sphysics.math.sum(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.abs(s-np.sum(self._array_np)).numpy(), 0., 6)


	def test_einsum(self):
		equation='i,j,j->i'
		correct = np.einsum(equation, self._array_np, self._array_np, self._array_np)
		s = sphysics.math.einsum(equation, self._array_np, self._array_np, self._array_np)
		self.assertIsInstance(s, np.ndarray)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-correct)), 0., 6)

		s = sphysics.math.einsum(equation, self._array_tf, self._array_tf, self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-correct)).numpy(), 0., 6)


	def test_transpose(self):
		twoD = self._array_np.reshape(2,5,-1)
		twoD_tf = tf.constant(twoD)

		s = sphysics.math.transpose(twoD,(1,0,2))
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s[1,...]-twoD[:,1,:])), 0., 6)

		s = sphysics.math.transpose(twoD_tf,(1,0,2))
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s[1,...]-twoD[:,1,:])).numpy(), 0., 6)


	def test_conjugate(self):
		s = sphysics.math.conjugate(self._array_np + 3j)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.conjugate(self._array_np+3j))), 0., 6)

		s = sphysics.math.conjugate(tf.cast(self._array_tf, tf.complex128) + 3j)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.conjugate(self._array_np+3j))).numpy(), 0., 6)


	def test_copy(self):
		s = sphysics.math.copy(self._array_np)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-self._array_np)), 0., 6)

		s = sphysics.math.copy(self._array_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-self._array_np)).numpy(), 0., 6)


	def test_cross(self):
		a = np.linspace(0, 10, 30).reshape((10,3))
		b = np.linspace(-5, 10, 30).reshape((10,3))
		a_tf = tf.constant(a)
		b_tf = tf.constant(b)

		s = sphysics.math.cross(a,b)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.cross(a,b))), 0., 6)

		s = sphysics.math.cross(a_tf, b_tf)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.cross(a,b))).numpy(), 0., 6)

		s = sphysics.math.cross(a.T,b.T, axis=0)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.cross(a.T,b.T, axis=0))), 0., 6)

		s = sphysics.math.cross(tf.transpose(a_tf), tf.transpose(b_tf), axis=0)
		self.assertIsInstance(s, tf.Tensor)
		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(s-np.cross(a.T,b.T, axis=0))).numpy(), 0., 6)




if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
