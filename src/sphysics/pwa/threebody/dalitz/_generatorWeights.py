# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Helper functions to calculate the generator weights of MC events generated with EvtGen, Created on Monday 30 01 2023
'''

from __future__ import absolute_import, print_function, division, annotations

import scipy.integrate

from .... import math
from .... import kinematics
from .... import amplitudes
from .... import lorentz


def integratePS(m123, m1, m2, m3):
	'''
	Integrate over breitWigner amplitude in the (12) system
	'''
	def integrant(m12):
		return 1. \
			 * kinematics.twobodyBreakupmomentumZeroBelowZero(m123**2, m12, m3) \
			 * kinematics.twobodyBreakupmomentumZeroBelowZero(m12**2, m1, m2)

	return scipy.integrate.quad(integrant, m1+m2, m123-m3)

def integrateBW(m123, m1, m2, m3, L, m0, g0):
	'''
	Integrate over breitWigner amplitude in the (12) system
	'''
	def integrant(m12):
		return math.abs2(amplitudes.resonance.breitWigner(m12, m1, m2, L, m0, g0)) \
			 * kinematics.twobodyBreakupmomentumZeroBelowZero(m123**2, m12, m3) \
			 * kinematics.twobodyBreakupmomentumZeroBelowZero(m12**2, m1, m2)

	return scipy.integrate.quad(integrant, m1+m2, m123-m3)


def dalitzNormalizedBreitWigner(m12, m123, m1, m2, m3, L, m0, g0):
	integral, integralUnc = integrateBW(m123, m1, m2, m3, L, m0, g0)
	if integralUnc > 1e-7*integral:
		raise Exception(f"integral={integral}; integralUncertainty={integralUnc} is too large")
	return amplitudes.resonance.breitWigner(m12, m1, m2, L, m0, g0)/math.sqrt(integral)


def calculateGeneratorWeightsB0ToKpipi0(pK, pPi, pPi0):
	"""Calculate the generator weights that were used to generate :math:`B_0 \\to K^{\\pm} \\pi^{\\pm} \\pi_0` events

	These events are not phase-space generated, but generated with some importance-sampling weights

	Returns:
		np.ndarray: Array of weights of the shape [<nEvents>]
	"""

	pKpi = pK + pPi
	mKpi = math.sqrt(lorentz.lp(pKpi, pKpi))
	pKpi0 = pK + pPi0
	mKpi0 = math.sqrt(lorentz.lp(pKpi0, pKpi0))
	pPipi0 = pPi + pPi0
	mPipi0 = math.sqrt(lorentz.lp(pPipi0, pPipi0))


	# only Kpi0
	# From BASF2
	mB0=5279.65e-3
	mK=493.677e-3
	mPi=139.57039e-3
	mPi0=134.9768e-3
	m0 = 1.1
	g0 = 1.0
	# phspFraction = 0.04+0.000002800 # the latter one comes from non-signal B0 decays
	phspFraction = 0.04

	generatorWeights = 0.32* math.abs2(dalitzNormalizedBreitWigner(mKpi,   mB0, mK,   mPi,  mPi0, 0, m0, g0)) \
					 + 0.32* math.abs2(dalitzNormalizedBreitWigner(mKpi0,  mB0, mK,   mPi0, mPi,  0, m0, g0)) \
					 + 0.32* math.abs2(dalitzNormalizedBreitWigner(mPipi0, mB0, mPi,  mPi0, mK,   0, m0, g0)) \
					 + phspFraction / integratePS(mB0, mK, mPi, mPi0)[0]
	return generatorWeights

def calculateGeneratorWeightsBpToKSpipi0(pKS, pPi, pPi0):
	"""Calculate the generator weights that were used to generate :math:`B^+ \\to K_S pi^+ pi^0` events

	These events are not phase-space generated, but generated with some importance-sampling weights

	Returns:
		np.ndarray: Array of weights of the shape [<nEvents>]
	"""

	pKSpi = pKS + pPi
	mKSpi = math.sqrt(lorentz.lp(pKSpi, pKSpi))
	pKSpi0 = pKS + pPi0
	mKSpi0 = math.sqrt(lorentz.lp(pKSpi0, pKSpi0))
	pPipi0 = pPi + pPi0
	mPipi0 = math.sqrt(lorentz.lp(pPipi0, pPipi0))


	# only Kpi0
	# From BASF2
	mBp=5279.34e-3
	mKS=497.611e-3
	mPi=139.57039e-3
	mPi0=134.9768e-3
	m0 = 1.1
	g0 = 1.0
	# phspFraction = 0.04+0.000002800 # the latter one comes from non-signal B0 decays
	phspFraction = 0.04

	generatorWeights = 0.32* math.abs2(dalitzNormalizedBreitWigner(mKSpi,   mBp, mKS,   mPi,  mPi0, 0, m0, g0)) \
					 + 0.32* math.abs2(dalitzNormalizedBreitWigner(mKSpi0,  mBp, mKS,   mPi0, mPi,  0, m0, g0)) \
					 + 0.32* math.abs2(dalitzNormalizedBreitWigner(mPipi0, mBp, mPi,  mPi0, mKS,   0, m0, g0)) \
					 + phspFraction / integratePS(mBp, mKS, mPi, mPi0)[0]
	return generatorWeights
