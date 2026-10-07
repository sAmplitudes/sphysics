# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Phase-space generators, Created on Monday 14 03 2022
'''

from __future__ import absolute_import, print_function, division

import numpy as np

from .. import math
from .. import kinematics
from .. import lorentz
from .. import random



generator = random.generators.phaseSpace


def genThreeVec(absVal):
	"""
	Generated a three vector (isotropic)
	@param absVal magnitude of the vector
	"""
	phi    = generator.uniform(-np.pi, np.pi, absVal.size)
	cost   = generator.uniform(-1., 1., size=absVal.size)
	sint   = math.sqrt(1.-cost**2)
	retVal = np.empty((3, absVal.size), dtype=absVal.dtype)
	retVal[0] = absVal*math.cos(phi)*sint
	retVal[1] = absVal*math.sin(phi)*sint
	retVal[2] = absVal*cost
	return retVal



def genFourVecFromEp(E,p):
	"""
	Generates isotropically distributed 4-vectors (mass = (E**2 - p**2)**.5)
	@param E energy of the four-vector
	@param p magnitue of the momentum of the four vector
	"""
	retVal      = np.empty((4,E.size))
	retVal[1:4] = genThreeVec(p)
	retVal[0]   = E
	return retVal

def genFourVecFrompm(p,m):
	"""
	Generates isotropically distributed 4-vectors (energy = (p**2 + m**2)**.5)
	@param p magnitue of the momentum of the four vector
	@param m mass of the four vector
	"""
	E = math.sqrt(p**2 + m**2)
	return genFourVecFromEp(E,p)


def twoBodyDecay(M, m1, m2):
	"""
	Generates isotropically distributed four-momentum arrays (:math:`E = \\sqrt{p^2 + m^2}`) of a two body decay.
	The number of events is given by the length of the mass-arrays.

	:param M: Invariant mass of the two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	"""
	absMom = kinematics.twobodyBreakupmomentum(M**2,m1,m2)
	p1 = genFourVecFrompm(absMom, m1)
	p2 = -p1[:]
	p2[0] = math.sqrt(p2[1]**2 + p2[2]**2 + p2[3]**2 + m2**2)
	return p1, p2


class kinWeightNbody(object):
	def __init__(self, mMother, fsMasses):
		"""
		Kinematic weight function for an N-body final state.
		Works only for nFS > 2
		@param mMother  mother mass
		@param fsMasses masses of final state particles [m1, ..., mN]
		"""
		self.mMother  = mMother
		self.nFS      = len(fsMasses)
		if self.nFS < 3:
			raise ValueError("Only works for nFS > 2.")
		self.fsMasses = fsMasses
		mSum = 0.
		for m in fsMasses:
			mSum += m
		if mSum > mMother:
			raise ValueError("Sum of fsMasses exceeds mother mass")

	def __call__(self, intermediaryMasses):
		"""
		@param intermediaryMasses numpy array with size (nFS-2, nEvents). One line is [m01, m012, m0123, ..., m012...N]
		"""
		allowed = intermediaryMasses[0] >=  self.fsMasses[0] + self.fsMasses[1]
		weight = math.sqrt(kinematics.twobodyBreakupmomentumSquared(intermediaryMasses[0]**2, self.fsMasses[0], self.fsMasses[1]),
		                   where=allowed, whereNotValue=0.0)
		for i in range(self.nFS - 3):
			allowed = intermediaryMasses[i+1] >= intermediaryMasses[i] + self.fsMasses[i+2]
			weight *= math.sqrt(kinematics.twobodyBreakupmomentumSquared(intermediaryMasses[i+1]**2, intermediaryMasses[i], self.fsMasses[i+2]),
			                    where=allowed, whereNotValue=0.)
		allowed = self.mMother >= intermediaryMasses[self.nFS-3] + self.fsMasses[self.nFS-1]
		weight *= math.sqrt(kinematics.twobodyBreakupmomentumSquared(self.mMother**2,intermediaryMasses[self.nFS-3], self.fsMasses[self.nFS-1]),
	                        where=allowed, whereNotValue= 0.)
		return weight


def genPointsND(nAttempts, distND, rangesMin, rangesMax, dim):
	"""
	Generates points according to a n-dimansional distribution
	@param n          number of attempts != number of points to be returned
	@param distND     distribution
	@param rangesMin  lower bounds
	@param rangesMax  upper bounds
	@param dim        number of dimensions
	"""
	pts = generator.random((dim,nAttempts))
	pts[:] *= (rangesMax - rangesMin)
	pts[:] += rangesMin
	weights = distND(pts)
	wMax = math.max(weights)
	return wMax, pts[...,weights > wMax * generator.random(nAttempts)]


def genNpointsND(nPoints, distND, rangesMin, rangesMax, dim):
	"""
	Generated n dim-dimensional points distributed according to distND
	@param n          number of points to be returned
	@param distND     distribution
	@param rangesMin  lower bounds
	@param rangesMax  upper bounds
	@param dim        number of dimensions
	"""
	retVal = np.empty((dim,nPoints))
	wMax, firstPoints = genPointsND(nPoints, distND, rangesMin, rangesMax, dim)
	nFound = firstPoints.shape[-1]
	retVal[...,:nFound] = firstPoints
	while nFound < nPoints:
		wMaxNew, newPoints = genPointsND(nPoints, distND, rangesMin, rangesMax, dim)
		if wMaxNew > wMax:
			reducedPoints       = retVal[...,:nFound][...,wMax/wMaxNew > generator.random(nFound)]
			nFound              = reducedPoints.shape[-1]
			retVal[...,:nFound] = reducedPoints
			wMax = wMaxNew
			nSet = min(newPoints.shape[-1], nPoints - nFound)
			retVal[...,nFound:nFound + nSet] = newPoints[...,:nSet]
		else:
			reducedPoints = newPoints[...,wMaxNew/wMax > generator.random(newPoints.shape[-1])]
			nSet = min(nPoints - nFound, reducedPoints.shape[-1])
			retVal[...,nFound:nFound+nSet] = reducedPoints[...,:nSet]
			nFound += nSet
	return retVal


class NBodyGenerator(object):
	"""
	Generates four-momentum arrays of n-body decays of one mother particle into n final-state particles.

	:param M: Mass of the mother particle.
	:param fsMasses: List of masses of the final-state particles.
	:param limitMasses: Limit mass of system (k) by limitMasses[k], Each key :math:`k` corresponds to a subsystem, where:

		- :math:`k=0 \\to (1,2)`
		- :math:`k=1 \\to (1,2,3)`
		- :math:`k=2 \\to (1,2,3,4)`, ...
	"""
	def __init__(self, M, fsMasses, limitMasses: dict=None):
		"""
		@param M        mother mass
		@param fsMasses list of final state masses
		@param limitMasses: Limit mass of system (k) by limitMasses[k], where k=  0->12, 1->123, 2->1234, ...
		"""
		self.M = M
		self.fsMasses = fsMasses
		self.nFsParticles = len(fsMasses)
		self.massWeightFunction = kinWeightNbody(M,self.fsMasses)
		lowerLims = [fsMasses[0] + fsMasses[1]]
		for i in range(2, len(fsMasses)-1):
			lowerLims.append(lowerLims[~0] + fsMasses[i])
		self.rangeMin = np.array(lowerLims).reshape(-1,1)
		upperLims = [M - fsMasses[~0]]
		for i in range(1, len(fsMasses)-2):
			upperLims.append(upperLims[~0] - fsMasses[~i])
		upperLims.reverse()
		self.rangeMax = np.array(upperLims).reshape(-1,1)
		if limitMasses is not None:
			for k, limits in limitMasses.items():
				if limits[0] < self.rangeMin[k,0]:
					raise Exception(f"New lower mass limit for k={k} would be lower than the kinematic limit!")
				self.rangeMin[k,0] = limits[0]
				if limits[1] > self.rangeMax[k,0]:
					raise Exception(f"New upper masss limit for k={k} would be larger than the kinematic limit!")
				self.rangeMax[k,0] = limits[1]


	def gen(self, nEvents):
		"""
		Generates `nEvents` samples of final-state four-momentum arrays for an n-body decay.
		Intermediate two-body decays are computed and Lorentz boosts are applied.

		:param nEvents: Number of decay events
		"""
		masses = genNpointsND(nEvents, self.massWeightFunction, self.rangeMin, self.rangeMax, self.nFsParticles - 2)
		# nGenerated=0
		# while nGenerated < nEvents:
		# 	masses = genNpointsND(nEvents-nGenerated, self.massWeightFunction, self.rangeMin, self.rangeMax, self.nFsParticles - 2)
		# 	if self.limitMasses:
		# 		iSelect = np.ones(masses.shape[-1])
		# 		for k, limits in self.limitMasses:
		# 			iSelect &= (masses[k] >= limits[0]) & (masses[k] < limits[1])
		# 		masses = masses[:,iSelect]
		# 	nGenerated = masses.shape[-1]
		retVal = [np.empty((4,nEvents)) for _ in range(self.nFsParticles)]
		retVal[0], retVal[1] = twoBodyDecay(masses[0], self.fsMasses[0], self.fsMasses[1])
		for i in range(1,self.nFsParticles - 1):
			if i < self.nFsParticles-2:
				mIsobar = masses[i]
			else:
				mIsobar = self.M
			pIsob, retVal[i+1] = twoBodyDecay(mIsobar, masses[i-1], self.fsMasses[i+1])
			boost = lorentz.getBoostFromRestFrame(pIsob)
			for j in range(i+1):
				retVal[j] = math.einsum('ije, je->ie', boost, retVal[j])
		return retVal
