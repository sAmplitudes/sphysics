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


class DiracTest(unittest.TestCase):
	def __init__(self, methodName: str = ...) -> None:
		super().__init__(methodName)

		self.dirac = sphysics.dirac.Weyl()
		pTau, pNu, p1, p2, p3 = sphysics.generators.tauToThreePi(1e6, sphysics.Constants.M.upsilon4S**2)
		self.pTau = pTau
		self.pNu = pNu

	def test_spinors(self):
		dirac = self.dirac
		pTau = self.pTau
		pNu = self.pNu
		# test normalization
		for isUpA in [True, False]:
			for isParticle in [True, False]:
				sA = dirac.getParticleSpinor(pTau, isUpA, isParticle)
				for isUpB in [True, False]:
					sB = dirac.getParticleSpinor(pTau, isUpB, isParticle)
					n = sphysics.math.einsum('ie,ie->e', dirac.bar(sA), sB)
					if isUpA != isUpB:
						expected = 0.
					elif isParticle:
						expected = 2*sphysics.Constants.M.tau
					else:
						expected = -2.*sphysics.Constants.M.tau
					d = np.max(np.abs( n  - expected))
					self.assertAlmostEqual(d, 0, 10)

		# test chirality of spin-less particles
		for isUpA in [True, False]:
			for isParticle in [True, False]:
				sA = dirac.getParticleSpinor(pNu, isUpA, isParticle)
				leftChiral = sphysics.math.einsum('ij,je->ie', dirac.oneMinusGamma5, sA)
				if (isUpA and isParticle) or (not isUpA and not isParticle): # left-chiral part should be zero
					self.assertAlmostEqual(np.max(np.abs(leftChiral)), 0, 6)

	def test_current(self):
		self.assertAlmostEqual(np.max(np.abs(self.dirac.weakLepCurr_particledecay(self.pTau, self.pNu, upTau=False, upNu=True))), 0., 6)
		self.assertAlmostEqual(np.max(np.abs(self.dirac.weakLepCurr_antiparticledecay(self.pTau, self.pNu, upTau=False, upNu=False))), 0., 6)

	def test_tensor(self):
		dirac = self.dirac
		up = dirac.weakLepCurr_particledecay(self.pTau, self.pNu, True, False)
		down = dirac.weakLepCurr_particledecay(self.pTau, self.pNu, False, False)
		tensor = (sphysics.lorentz.tensorproduct(up, np.conj(up)) + sphysics.lorentz.tensorproduct(down, sphysics.math.conjugate(down)))/2
		self.assertAlmostEqual(np.max(np.abs(tensor - dirac.weakLepTensor_unpolarizedDecay(self.pTau, self.pNu, True))), 0, 10)

		up = dirac.weakLepCurr_antiparticledecay(self.pTau, self.pNu, True, True)
		down = dirac.weakLepCurr_antiparticledecay(self.pTau, self.pNu, False, True)
		tensor = (sphysics.lorentz.tensorproduct(up, np.conj(up)) + sphysics.lorentz.tensorproduct(down, sphysics.math.conjugate(down)))/2
		self.assertAlmostEqual(np.max(np.abs(tensor - dirac.weakLepTensor_unpolarizedDecay(self.pTau, self.pNu, False))), 0, 10)


if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
