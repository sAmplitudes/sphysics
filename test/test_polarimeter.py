#!/usr/bin/env python
# coding: utf-8
'''


@author: Stefan Wallner
'''

from __future__ import absolute_import, print_function, division

import unittest

import sphysics
from sphysics import Constants as C
from sphysics import lorentz
from sphysics.pwa.polarimeter import taupair

import numpy as np


# Spin-density parameters of the toy samples in the (n, r, k) component order of the nrk frame.
# They are chosen such that 0 < 1 + B+.h+ + B-.h- + h+.C.h- < 2 for all unit vectors h+, h-.
B_POS = np.array([0.1, -0.1, 0.05])
B_NEG = np.array([0.0, 0.1, -0.05])
C_IJ = np.array([[-0.5, 0.1,  0.0],
                 [0.05, 0.4, -0.1],
                 [0.0, 0.05,  0.3]])


def _isotropicUnitVectors(rng, nEvents):
	'''Draw isotropically distributed 3D unit vectors of shape ``(3, nEvents)``.'''
	vec = rng.normal(size=(3, nEvents))
	return vec/np.linalg.norm(vec, axis=0)


def _generateSpinCorrelatedPolarimeters(rng, nEvents):
	'''Draw polarimeter 4-vectors (zero time component) of shape ``(4, nAccepted)`` for tau+ and tau- following
	1 + B+.h+ + B-.h- + h+.C.h- by accept-reject on isotropic unit vectors.'''
	h_pos = _isotropicUnitVectors(rng, nEvents)
	h_neg = _isotropicUnitVectors(rng, nEvents)
	weight = 1. + B_POS @ h_pos + B_NEG @ h_neg + np.einsum('in,ij,jn->n', h_pos, C_IJ, h_neg)
	accepted = rng.uniform(0., 2., nEvents) < weight
	zeros = np.zeros((1, np.sum(accepted)))
	return np.vstack((zeros, h_pos[:, accepted])), np.vstack((zeros, h_neg[:, accepted]))


def _rhoCurrent(pPi, pPi0):
	'''Hadronic current of tau -> rho nu up to a Lorentz-invariant factor, i.e. the pi pi0 relative momentum
	projected transverse to the rho momentum.'''
	q = pPi + pPi0
	d = pPi - pPi0
	return d - lorentz.lp(q, d)/lorentz.lp(q, q)*q


class PolarimeterTest(unittest.TestCase):
	'''Tests for the Fano coefficient extraction in sphysics.pwa.polarimeter.taupair.'''

	def _generateTauPairSample(self, nEvents, withPi0):
		'''Generate tau+ tau- -> pi(pi0) nu pairs at rest in the e+e- CMS, where the e+ is along +z, and boost the
		event along +z into a Belle II like lab frame.

		:return: The sample and the positron 4-momentum in the lab frame and the expected polarimeter vectors of shape
		         ``(4, nEvents)`` in the nrk frame
		'''
		pTau_pos, pTau_neg = sphysics.generators.twoBodyDecay(np.full(nEvents, C.M.upsilon_4S_0), C.M.tau, C.M.tau)
		pTauRest = np.zeros((4, nEvents))
		pTauRest[0] = C.M.tau

		# nrk axes in the CMS: k along the tau+, r in the plane spanned by the tau+ and the e+ beam
		k = pTau_pos[1:]/np.linalg.norm(pTau_pos[1:], axis=0)
		beam = np.array([0., 0., 1.]).reshape(3, 1)
		cosTheta = k[2]
		sinTheta = np.sqrt(1. - cosTheta**2)
		r = (beam - cosTheta*k)/sinTheta
		n = np.cross(beam, k, axis=0)/sinTheta

		betaGammaLab = 0.28
		pLab = np.tile(np.array([C.M.upsilon_4S_0*np.sqrt(1. + betaGammaLab**2), 0., 0., C.M.upsilon_4S_0*betaGammaLab]).reshape(4, 1), (1, nEvents))
		boostLab = lorentz.getBoostFromRestFrame(pLab)
		pPositron = np.tile(np.array([C.M.upsilon_4S_0/2., 0., 0., np.sqrt(C.M.upsilon_4S_0**2/4. - C.M.e**2)]).reshape(4, 1), (1, nEvents))
		pPositron = lorentz.applyBoost(boostLab, pPositron)

		sample = sphysics.eventselection.Variables()
		expectedH = {}
		for charge, pTau in (('pos', pTau_pos), ('neg', pTau_neg)):
			if withPi0:
				pPi, pPi0, pNu = sphysics.generators.NBodyGenerator(C.M.tau, [C.M.pi, C.M.pi0, 0.0]).gen(nEvents)
				current = _rhoCurrent(pPi, pPi0)
			else:
				pPi, pNu = sphysics.generators.twoBodyDecay(np.full(nEvents, C.M.tau), C.M.pi, 0.0)
				current = pPi
			isParticle = np.full(nEvents, charge == 'neg')
			hRest = taupair.get_polarimeter_from_J(current, isParticle, pTauRest, pNu)[1:]

			# The nrk frame differs from the tau rest frame reached directly from the CMS only by the rotation to the (n, r, k) axes
			expectedH[charge] = np.vstack((np.zeros(nEvents), np.sum(n*hRest, axis=0), np.sum(r*hRest, axis=0), np.sum(k*hRest, axis=0)))

			boostTau = lorentz.getBoostFromRestFrame(pTau)
			momenta = {'tau': pTau, 'nu': lorentz.applyBoost(boostTau, pNu), 'pi': lorentz.applyBoost(boostTau, pPi)}
			if withPi0:
				momenta['pi0'] = lorentz.applyBoost(boostTau, pPi0)
			for particle, p in momenta.items():
				name = f'{particle}_{charge}'
				sample.addVariable(name, auto=False, is4Momentum=True)
				sample[name] = lorentz.applyBoost(boostLab, p)

		return sample, pPositron, expectedH


	def _checkFanoCoeffsFrame(self, calculatePolarimeterVectors, calculateFanoCoeffs, withPi0):
		'''Check the polarimeter vectors against the expected ones in the nrk frame and check that the returned Fano
		coefficients and their uncertainties are the projections of these polarimeter vectors.'''
		sample, pPositron, expectedH = self._generateTauPairSample(10_000, withPi0)

		h_pos, h_neg = calculatePolarimeterVectors(sample, pPositron)

		for charge, h in (('pos', h_pos), ('neg', h_neg)):
			self.assertEqual(h.shape, (4, sample.nEvents))
			self.assertAlmostEqual(np.max(np.abs(h[0])), 0., 8)
			self.assertAlmostEqual(np.max(np.abs(np.sum(h[1:]**2, axis=0) - 1.)), 0., 8)
			self.assertAlmostEqual(np.max(np.abs(h - expectedH[charge])), 0., 8)

		fanoCoeffs = calculateFanoCoeffs(sample, pPositron)
		fanoCoeffsRef = taupair.calculateFanoCoeffsProjection(expectedH['pos'], expectedH['neg'])
		self.assertEqual(len(fanoCoeffs), 6)
		for value, ref in zip(fanoCoeffs, fanoCoeffsRef):
			self.assertEqual(value.shape, ref.shape)
			self.assertAlmostEqual(np.max(np.abs(value - ref)), 0., 8)


	def test_fanoCoeffsProjectionClosure(self):
		'''Extract the Fano coefficients from a toy sample generated with known B+, B-, and C and check that they agree
		with the generated values within their uncertainties.'''
		rng = np.random.default_rng(1234)
		h_pos, h_neg = _generateSpinCorrelatedPolarimeters(rng, 2_000_000)

		b_pos, b_neg, c_ij, b_pos_unc, b_neg_unc, c_ij_unc = taupair.calculateFanoCoeffsProjection(h_pos, h_neg)

		self.assertEqual(b_pos.shape, (3, 1))
		self.assertEqual(b_neg.shape, (3, 1))
		self.assertEqual(c_ij.shape, (3, 3))
		self.assertEqual(b_pos_unc.shape, (3, 1))
		self.assertEqual(b_neg_unc.shape, (3, 1))
		self.assertEqual(c_ij_unc.shape, (3, 3))

		estimates = np.concatenate((b_pos.ravel(), b_neg.ravel(), c_ij.ravel()))
		unc = np.concatenate((b_pos_unc.ravel(), b_neg_unc.ravel(), c_ij_unc.ravel()))
		truth = np.concatenate((B_POS, B_NEG, C_IJ.ravel()))
		pulls = (estimates - truth)/unc
		self.assertLess(np.max(np.abs(pulls)), 8.)


	def test_fanoCoeffsProjectionUncertainties(self):
		'''Check the calibration of the uncertainties of the Fano coefficients using the pull distribution of many
		independent toy samples.'''
		rng = np.random.default_rng(4321)
		truth = np.concatenate((B_POS, B_NEG, C_IJ.ravel()))
		pulls = []
		for _ in range(50):
			h_pos, h_neg = _generateSpinCorrelatedPolarimeters(rng, 40_000)
			b_pos, b_neg, c_ij, b_pos_unc, b_neg_unc, c_ij_unc = taupair.calculateFanoCoeffsProjection(h_pos, h_neg)
			estimates = np.concatenate((b_pos.ravel(), b_neg.ravel(), c_ij.ravel()))
			unc = np.concatenate((b_pos_unc.ravel(), b_neg_unc.ravel(), c_ij_unc.ravel()))
			pulls.append((estimates - truth)/unc)
		pulls = np.array(pulls)

		self.assertLess(np.abs(np.mean(pulls)), 0.3)
		self.assertLess(np.abs(np.std(pulls) - 1.), 0.3)


	def test_fanoCoeffsPiPiFrame(self):
		'''Check the polarimeter vectors of tau+ tau- -> pi+ pi- nu nu in the nrk frame of the tau+.'''
		self._checkFanoCoeffsFrame(taupair.calculatePolarimeterVectorPiPi, taupair.calculateFanoCoeffsPiPi, withPi0=False)


	def test_fanoCoeffsRhoRhoFrame(self):
		'''Check the polarimeter vectors of tau+ tau- -> rho+ rho- nu nu in the nrk frame of the tau+.'''
		self._checkFanoCoeffsFrame(taupair.calculatePolarimeterVectorRhoRho, taupair.calculateFanoCoeffsRhoRho, withPi0=True)




if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
