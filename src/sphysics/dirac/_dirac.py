# coding: utf-8
'''
:Author: Stefan Wallner

:Description: Formalsim for dirac particles, Created on Tuesday 15 03 2022
'''

from __future__ import absolute_import, print_function, division

import numpy as np

from .. import math
from .. import lorentz

unit2 = np.array([[1.,0.],[0.,1.]], dtype = complex) # Two-dimensional unit matrix
zero2 = np.zeros((2,2), dtype = complex) # Two-dienstional zero-matrix
unit4 = np.block([[unit2, zero2],[zero2,unit2]]) #Four-dimensional unit matrix


sigma = np.array([
	[[0., 1. ],[1.,0. ]],
	[[0.,-1.j],[1.j,0.]],
	[[1.,  0.],[0.,-1.]]], dtype = complex) # Two-dimensional Pauli matrices




class Formalism(object):
	def __init__(self) -> None:
		self.gamma = None
		self.gamma5  = None
		self.oneMinusGamma5 = None
		self.gammaMuGamma5 = None
		self.gammaMuOneMinusGamma5 = None
		self.sigmaMuNu = None
		self.sigmaMuNuGamma5 = None

	def _initGammaMatrices(self) -> None:
		self.gamma5 = 1j*np.dot(np.dot(self.gamma[0], self.gamma[3]), np.dot(self.gamma[1], self.gamma[2]))
		self.gammaMuGamma5 = np.array([np.dot(self.gamma[m], self.gamma5) for m in range(4)])
		self.oneMinusGamma5 = unit4 - self.gamma5
		self.gammaMuOneMinusGamma5 = np.array([np.dot(self.gamma[m], self.oneMinusGamma5) for m in range(4)])

		self.sigmaMuNu = np.zeros((4,4,4,4), dtype = complex) # Commutator of two gamma matrices (as used in EDm and MDM)
		self.sigmaMuNuGamma5 = np.zeros((4,4,4,4), dtype = complex)
		for m in range(4):
			for n in range(4):
				self.sigmaMuNu[m,n] = 1.j*self.commutator(self.gamma[m], self.gamma[n])/2
				self.sigmaMuNuGamma5[m,n] = np.dot(self.sigmaMuNu[m,n], self.gamma5)

	def commutator(self,a,b):
		'''Commutator of two matrices

		:param a: Matrix a
		:param b: Matrix b
		:return: The commutator of the two matrices
		'''
		return np.dot(a,b) - np.dot(b,a)


	def bar(self, u):
		'''Computes the Dirac adjoint of spinor u.

		:param u: The spinor u
		:type u: np.ndarray
		:return: Dirac adjoint of the spinor
		'''
		return math.einsum('i...,ij->j...', math.conjugate(u), self.gamma[0])


	def weakLepCurr_particledecay(self, pTau, pNu, upTau, upNu):
		"""
		Constructs the leptonic current  tau-decay events
		Four-momenta are :math:`[E, p_x, p_y, p_z]` for every event

		:param pTau: Four-momentum of the decaying :math:`\\tau^{-}`.
		:param pNu: Four-momentum of the final-state :math:`\\nu`.
		:param upTau: Helicity flags for the :math:`\\tau^{-}` (True = up, False = down).
		:param upNu: Helicity flags for :math:`\\nu` (should always be False, except when :math:`p_{\\nu}^2 \\neq 0.`, i.e., when the neutrino is massive).
		"""
		uTau = self.getParticleSpinor(pTau, upTau, True)
		uNu  = self.getParticleSpinor(pNu, upNu, True)

		# calculates \bar{u_nu} \gamma^mu (1-\gamma^5) u_tau, where \bar{u} = s^\dagger \gamma^0
		return np.einsum('ie,ij,mjk,ke->me',math.conjugate(uNu),self.gamma[0],self.gammaMuOneMinusGamma5,uTau)


	def weakLepCurr_antiparticledecay(self, pTau, pNu, upTau, upNu):
		"""
		Constructs the leptonic current for an array of anti-tau-decay events
		Four-momenta are :math:`[E, p_x, p_y, p_z]` for every event

		:param pTau: Four-momenta of the decaying :math:`\\tau^{+}`
		:param pNu: Four-momenta of the final-state :math:`\\nu`
		:param upTau: Helicity flags for the :math:`\\tau^{+}` (True = up, False = down)
		:param upNu: Helicity flags for the nu (should always be False, except :math:`p_{\\nu}^2 \\neq 0.`, i.e., when pNu is massive)
		"""
		vTau = self.getParticleSpinor(pTau, upTau, False)
		vNu  = self.getParticleSpinor(pNu, upNu, False)
		return np.einsum('ie,ij,mjk,ke->me',math.conjugate(vTau),self.gamma[0],self.gammaMuOneMinusGamma5,vNu)

	def weakLepTensor_unpolarizedDecay(self, pTau, pNu, isParticle):
		"""
		:math:`L_{\\mu\\nu} = \\frac{1}{2} \\sum_h l^\\mu_h (l_h^\\nu)^*`

		Four-momenta are :math:`[E, p_x, p_y, p_z]` for every event

		:param pTau: Four-momenta of the decaying :math:`\\tau^{-}`
		:param pNu: Four-momenta of the final-state :math:`\\nu`
		"""
		fakk = math.full((1,pTau.shape[-1]), 1., pTau)
		if isParticle is False:
			fakk *= -1.
		if not np.isscalar(isParticle):
			fakk[:,~isParticle] = -1.
		retVal = 4.j * fakk * math.einsum('ijkl,ke,le,i,j->ije',lorentz.levitCivitaSymbol,pNu,pTau,lorentz.et,lorentz.et)
		retVal += 4. * math.einsum('ie,je->ije', pTau, pNu)
		retVal += 4. * math.einsum('ie,je->ije', pNu, pTau)
		retVal -= 4. * math.einsum('ie,ie,i,kl->kle', pTau, pNu, lorentz.et, lorentz.eta)
		return retVal

	def getParticleSpinor(self, p: np.ndarray, isUp: np.ndarray, isPart: np.ndarray) -> np.ndarray:
		"""
		Produces the spinors for particles in the Wely representation.
		One method for particles AND antiparticles
		@param p      four momenta as np.array with indices [m,e], where e = event index m = [E,x,y,z]
		@param isUp   flag, whether the spinor has helicity up
		@param isPart flag, whether it's a particle ir antiparticle
		"""
		raise NotImplementedError("getParticleSpinor not implemented.")


class Weyl(Formalism):

	def __init__(self) -> None:
		super().__init__()

		# gamma matrices
		# indices (m, i, j), where m is the lorentz index and i,j are the spinor indices
		self.gamma = np.array([np.block([[zero2, unit2],[unit2,zero2]])] + [np.block([[zero2, sigma[i]],[-sigma[i],zero2]]) for i in range(3)])
		self._initGammaMatrices()


	def getParticleSpinor_faib(self, p: np.ndarray, isUp: np.ndarray, isPart: np.ndarray) -> np.ndarray:
		"""
		Original function from Fabian. Potentially wrong definition of anti-particle spinors
		Produces the spinors for particles in the Wely representation.
		One method for particles AND antiparticles

		:param p: Four momenta as np.array with indices [m,e], where e = event index, m = [E, x, y, z]
		:param isUp: Flag indicating whether the spinor has helicity up
		:param isPart: Flag indicating whether it's a particle or antiparticle
		"""
		if np.isscalar(isUp):
			isUp = math.full(p.shape[-1], isUp, p, dtype = bool)

		if np.isscalar(isPart):
			isPart = math.full(p.shape[-1], isPart, p, dtype = bool)

		pAbs  = math.sqrt(math.sum(p[1:4]**2, axis = 0))
		energyPlusMom   = math.sqrt(p[0] + pAbs)
		energyMinusMom   = p[0] - pAbs
		below = energyMinusMom < 0.
		close = np.isclose(energyMinusMom[below]/energyPlusMom[below]/10. +1., 1.)
		if np.sum(~close) > 0:
			raise ValueError("ERROR: Found non-zero negative E-p in getParticleSpinor(...) for "+str(np.sum(~close))+" of "+str(p.shape[-1])+" events.")
		energyMinusMom[below] = 0.
		energyMinusMom  = math.sqrt(energyMinusMom)

		cost = math.full(p.shape[-1], 1., p)
		phi  = math.zeros(p.shape[-1], p)
		cosHalf = math.full(p.shape[-1],1., p)
		sinHalf = math.zeros(p.shape[-1], p)
		nonZero = pAbs/p[0] > 1e-10 # avoid numerical issues for very small momenta (w/r/t the energy), where phi and cost are not well defined

		phi[nonZero]     = math.arctan2(p[2,nonZero], p[1,nonZero])
		cost[nonZero]    = p[3,nonZero]/pAbs[nonZero]
		cosHalf[nonZero] = math.sqrt((1.+cost[nonZero])/2)
		sinHalf[nonZero] = math.sqrt((1.-cost[nonZero])/2)

		basisSpinor = math.empty((2, p.shape[-1]), p, dtype = complex)

		basisSpinor[0, isUp] = cosHalf[isUp]
		basisSpinor[1, isUp] = sinHalf[isUp] * math.exp( 1.j*phi[isUp])
		basisSpinor[0,~isUp] =-sinHalf[~isUp] * math.exp(-1.j*phi[~isUp])
		basisSpinor[1,~isUp] = cosHalf[~isUp]

		upperPreFactor = math.empty(p.shape[-1], p)
		lowerPreFactor = math.empty(p.shape[-1], p)

		upperPreFactor[ isUp] = energyMinusMom[ isUp]
		lowerPreFactor[ isUp] = energyPlusMom[ isUp]

		upperPreFactor[~isUp] = energyPlusMom[~isUp]
		lowerPreFactor[~isUp] = energyMinusMom[~isUp]

		upperPreFactor[~isPart] *= -1

		retVal = math.empty((4,p.shape[-1]), p, dtype = complex)
		retVal[0:2] = upperPreFactor[None,:] * basisSpinor
		retVal[2:4] = lowerPreFactor[None,:] * basisSpinor
		return retVal


	def getParticleSpinor(self, p: np.ndarray, isUp: np.ndarray, isPart: np.ndarray) -> np.ndarray:
		"""
		Produces the spinors for particles in the Wely representation.
		One method for particles AND antiparticles

		The spinors are normalized such that :math:`u_i \\cdot u_j = 2 m \\delta_{ij}`

		Spinor of anti-particles
		Applying :math:`\\gamma^5` to the a righ-helical particle spinor gives a left-helical anti-particle spinor.
		Hence, we construct in the following the particle spinor.
		In order to construct the anti-particle spinor, we construct the particle spinor with opposite helicity and multiply it by :math:`\\gamma`.

		If calculated in the rest frame of the particle, i.e., if p=0, the spinor is calculated for a particle going along the z-axis, i.e., :math:`\cos\theta=1` and :math:`\phi=0`.

		:param p: Four momenta as np.array with indices :math:`[m,e]`, where :math:`e` is the event index, and :math:`m = [E, x, y, z]`.
		:param isUp: Flag indicating whether the spinor has helicity up.
		:param isPart: Flag indicating whether it is a particle or an antiparticle.
		"""
		if np.isscalar(isUp):
			isUp = math.full(p.shape[-1], isUp, p, dtype = bool)
		else:
			isUp = math.copy(isUp) # we need to change it for anti particles

		if np.isscalar(isPart):
			isPart = math.full(p.shape[-1], isPart, p, dtype = bool)

		# invert helicity for anti-particles (see description)
		isUp[~isPart] = ~isUp[~isPart]

		pAbs  = math.sqrt(math.sum(p[1:4]**2, axis = 0))
		energyPlusMom   = p[0] + pAbs
		energyMinusMom  = p[0] - pAbs
		below = energyMinusMom < 0.
		close = np.isclose(energyMinusMom[below]/energyPlusMom[below]/10. +1., 1.)
		if np.sum(~close) > 0:
			raise ValueError("ERROR: Found non-zero negative E-p in getParticleSpinor(...) for "+str(np.sum(~close))+" of "+str(p.shape[-1])+" events.")
		energyMinusMom[below] = 0.

		cost = math.full(p.shape[-1], 1., p)
		phi  = math.zeros(p.shape[-1], p)
		nonZero = pAbs/p[0] > 1e-10 # avoid numerical issues for very small momenta (w/r/t the energy), where phi and cost are not well defined

		phi[nonZero]     = math.arctan2(p[2,nonZero], p[1,nonZero])
		cost[nonZero]    = p[3,nonZero]/pAbs[nonZero]
		cosHalf = math.sqrt((1.+cost)/2)
		sinHalf = math.sqrt((1.-cost)/2)

		basisSpinor = math.empty((2, p.shape[-1]), p, dtype = complex)

		basisSpinor[0, isUp] = cosHalf[isUp]
		basisSpinor[1, isUp] = sinHalf[isUp]  * math.exp( 1.j*phi[isUp])
		basisSpinor[0,~isUp] =-sinHalf[~isUp] * math.exp(-1.j*phi[~isUp])
		basisSpinor[1,~isUp] = cosHalf[~isUp]

		upperPreFactor = math.empty(p.shape[-1], p)
		lowerPreFactor = math.empty(p.shape[-1], p)

		upperPreFactor[ isUp] = math.sqrt(energyMinusMom[isUp])
		lowerPreFactor[ isUp] = math.sqrt(energyPlusMom[isUp])

		upperPreFactor[~isUp] = math.sqrt(energyPlusMom[~isUp])
		lowerPreFactor[~isUp] = math.sqrt(energyMinusMom[~isUp])

		upperPreFactor[~isPart] *= -1 # multiply anit-particle spinor by gamma^5 (see description)

		retVal = math.empty((4,p.shape[-1]), p, dtype = complex)
		retVal[0:2] = upperPreFactor[None,:] * basisSpinor
		retVal[2:4] = lowerPreFactor[None,:] * basisSpinor
		return retVal
