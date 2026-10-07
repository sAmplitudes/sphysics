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


class GeneratorsTest(unittest.TestCase):
	def test_generators(self):
		'''
		Test phase-space generators
		'''

		pTau, pNu, p1, p2, p3 = sphysics.generators.tauToThreePi(1e6, 10.5**2)

		p12 = p1+p2
		m12Squared = sphysics.lorentz.lp(p12,p12)

		p13 = p1+p3
		m13Squared = sphysics.lorentz.lp(p13,p13)

		p23 = p2+p3
		m23Squared = sphysics.lorentz.lp(p23,p23)

		pN1 = pNu+p1
		mN1Squared = sphysics.lorentz.lp(pN1,pN1)

		pN2 = pNu+p2
		mN2Squared = sphysics.lorentz.lp(pN2,pN2)

		pN3 = pNu+p3
		mN3Squared = sphysics.lorentz.lp(pN3,pN3)

		p123 = p1+p2+p3
		m123Squared = sphysics.lorentz.lp(p123,p123)
		m123 = sphysics.math.sqrt(m123Squared)


		pN12 = pNu+p1+p2
		mN12Squared = sphysics.lorentz.lp(pN12, pN12)
		mN12 = sphysics.math.sqrt(mN12Squared)

		pTot = p123+pNu

		self.assertAlmostEqual(sphysics.math.max(sphysics.math.abs(pTot -pTau)), 0., 12)

		iSelect = np.logical_and( m123 > 1.215, m123 < 1.22)

		self._checkDalitz(m12Squared, m13Squared, iSelect)
		self._checkDalitz(m12Squared, m23Squared, iSelect)
		self._checkDalitz(m13Squared, m23Squared, iSelect)
		self._checkDalitz(mN1Squared, mN2Squared, iselect=(mN12 > 1.215)*(mN12 < 1.22))

	def _checkDalitz(self, mas, mbs, iselect):
		counts, _, _ = np.histogram2d(mas[iselect], mbs[iselect], bins=30)
		var = np.var(counts[counts>5])
		mean = np.mean(counts[counts>5])
		self.assertLess(var, 1.2*mean)


if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
