#!/usr/bin/env python
# coding: utf-8
'''


@author: Stefan Wallner
'''

from __future__ import absolute_import, print_function, division

import unittest

import sphysics

import numpy as np
import tensorflow as tf


class TauPWATest(unittest.TestCase):
	'''Tests for the tau -> 3pi partial-wave-analysis helpers in sphysics.pwa.threebody.tau and sphysics.pwa.utils.'''

	def __init__(self, methodName: str = ...) -> None:
		'''Generate a shared sample of tau -> 3pi (+ nu) events used by all test methods.'''
		super().__init__(methodName)

		pTau, pNu, p1, p2, p3 = sphysics.generators.tauToThreePi(1e6, sphysics.Constants.M.upsilon4S**2)
		self.p1 = p1
		self.p2 = p2
		self.p3 = p3
		self.p123 = p1+p2+p3
		self.pTau = pTau
		self.pNu = pNu
		self.m123 = sphysics.math.sqrt(sphysics.lorentz.lp(self.p123,self.p123))


	def test_genTauMomentumComponentsCMS(self):
		'''Check the CMS decomposition of the tau momentum into a component parallel to p123 and two
		orthogonal components: verify the mutual orthogonality of the three components, that the
		orthogonal components are orthogonal to p123, and that the components reconstruct pTau.'''
		para, portho1, portho2 = sphysics.pwa.threebody.tau.getTauMomentumComponentsCMS(self.p123, sphysics.Constants.M.upsilon4S/2., sphysics.Constants.M.tau)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(portho1, portho2))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(portho1, self.p123))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(portho2, self.p123))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(self.pTau - para, self.p123))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(para, portho1))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(para, portho2))), 0., 10)

		cosAlpha = -sphysics.lorentz.lp(self.pTau-para, portho1)/(sphysics.math.sqrt(sphysics.math.sum(((self.pTau-para)**2)[1:],axis=0))*sphysics.math.sqrt(-sphysics.lorentz.lp(portho1,portho1)))
		cosBeta = -sphysics.lorentz.lp(self.pTau-para, portho2)/(sphysics.math.sqrt(sphysics.math.sum(((self.pTau-para)**2)[1:],axis=0))*sphysics.math.sqrt(-sphysics.lorentz.lp(portho2,portho2)))
		sinAlpha = cosBeta

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(self.pTau-para-cosAlpha*portho1, portho1))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(self.pTau-para-cosAlpha*portho1, para))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(self.pTau-para-sinAlpha*portho2, portho2))), 0., 10)

		self.assertAlmostEqual(np.max(np.abs(sphysics.lorentz.lp(self.pTau-para-sinAlpha*portho2, para))), 0., 10)

		reco = para + cosAlpha*portho1 + sinAlpha*portho2
		diff = self.pTau - reco
		self.assertAlmostEqual(np.max(np.abs(diff)), 0., 9)


	def test_calcAngles(self):
		'''Cross-check the closed-form tau helicity angle formulas against the generic isobar-tree
		helicity angle calculation, both in the tau rest frame and in the CMS, and confirm the tree
		angles are unchanged whether or not the neutrino is included as a fourth particle.'''
		anglesAll = sphysics.pwa.utils.calculateIsobarTreeHelicityAngles([self.p1, self.p2, self.p3, self.pNu])
		angles123 = sphysics.pwa.utils.calculateIsobarTreeHelicityAngles([self.p1, self.p2, self.p3])

		(cosT_tauRF, _), _ = sphysics.pwa.utils.calcHelicityAngles(self.pTau, self.p123, np.array([0.,0,0.,1.]))
		cosTAlt_tauRF = sphysics.pwa.threebody.tau.calcTauHelicityCosTheta_tauRF(self.pTau[0], sphysics.Constants.M.tau**2, self.p123[0], self.m123**2)

		self.assertAlmostEqual(np.max(np.abs(cosT_tauRF-cosTAlt_tauRF)), 0., 9)
		self.assertAlmostEqual(np.max(np.abs(cosT_tauRF-anglesAll['cosTheta_123__1234'])), 0., 9)
		self.assertAlmostEqual(np.max(np.abs(angles123['cosTheta_1__12']-anglesAll['cosTheta_1__12'])), 0., 9)

		cosT_CMS = np.sum( self.pTau[1:]*self.p123[1:], axis=0)/np.sqrt( np.sum(self.pTau[1:]**2,axis=0)*np.sum(self.p123[1:]**2,axis=0) )
		cosTAlt_CMS = sphysics.pwa.threebody.tau.calcTauHelicityCosTheta_CMS(self.pTau[0], sphysics.Constants.M.tau**2, self.p123[0], self.m123**2)

		self.assertAlmostEqual(np.max(np.abs(cosT_CMS-cosTAlt_CMS)), 0., 10)


	def test_calcgroupangles(self):
		'''Verify that calculateIsobarGroupHelicityAngles gives the same helicity angles as
		calculateIsobarTreeHelicityAngles for a boosted B -> 4pi phase-space sample, modulo the
		2pi ambiguity of the angle definition.'''
		gen = sphysics.generators.NBodyGenerator(sphysics.Constants.M.B, [sphysics.Constants.M.pi]*4)
		momenta = gen.gen(1_000_000)
		pR = np.array([(sphysics.Constants.M.B**2+1**2),1,0,0])[:,None]
		boost = sphysics.lorentz.getBoostFromRestFrame(pR)
		momenta = [ sphysics.lorentz.applyBoost(boost, m) for m in momenta ]
		anglesI = sphysics.pwa.utils.calculateIsobarTreeHelicityAngles(momenta, momentaInCMS=False)
		anglesII = sphysics.pwa.utils.calculateIsobarGroupHelicityAngles((1,2,3), momenta, momentaInCMS=False)

		for k in anglesI:
			d = anglesI[k] - anglesII[k]
			# handle 2pi ambiguity
			d[d > np.pi] -= 2*np.pi
			d[d<-np.pi] += 2*np.pi
			self.assertAlmostEqual(np.max(np.abs(d)), 0.0, 10)

if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
