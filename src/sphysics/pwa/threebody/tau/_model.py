# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Model for tau -> 3hadron decays, Created on Tuesday 15 03 2022
'''

from __future__ import absolute_import, print_function, division
from argparse import ArgumentError
from typing import Callable, Any


import numpy as np

from .... import dirac
from .... import math
from .... import lorentz
from .... import amplitudes
from ....utils import Logger
from ..._model import PWAModel
from .._hadronicTensors import HadronicTensors, Wave

from ._utils import decomposeHermitianMatrix2, EventsTau2ThreeChargedPi
from ._kinematics import getTauMomentumComponentsCMS


log = Logger("tau")




class Tau2ThreeCharedPi(PWAModel):
	'''
	PWA model for :math:`\\tau^{\\pm} \\to \\pi^{\\mp} \\pi^{\\pm} \\pi^{\\pm} \\nu_{\\tau}`.

	The order of the particles is as given by the above reaction,
	i.e. the first two pions have opposite charge and form the isobar system and the second and the last pions are exchanged for the bose symmetrization
	'''
	def __init__(self, name: str, description: str = None) -> None:
		super().__init__(name, description)
		self._hadronicTensors = HadronicTensors()
		self._dirac = dirac.Weyl()

		self._waveIdx2HadronicTensor = []

		self._dynamicIsobarAmplitudes = []
		self._waveIdx2DynamicIsobarAmplitudeIdx = []
		self._barrierIsobarCompensation = []
		self._waveIdx2BarrierIsobarCompensationIdx = []
		self._dynamicXAmplitudes = []
		self._waveIdx2DynamicXAmplitudeIdx = []
		self._barrierXCompensation = []
		self._waveIdx2BarrierXCompensationIdx = []
		self._decayAmplitudesIntegratedNormIntegralsForParticles = None  # normalization integrals for integrated decay amplitudes: shape (1, nWAves, 1)
		self._decayAmplitudesIntegratedNormIntegralsForAntiparticles = None  # normalization integrals for integrated decay amplitudes: shape (1, nWAves, 1)


	def addWave(self, waveName: str, hadronicTensorName: str, dynamicXAmplitude: Callable[[np.ndarray], np.ndarray], dynamicIsobarAmplitude: Callable[[np.ndarray], np.ndarray],
			    barrierXCompensation: Callable[[int, np.ndarray, np.ndarray], np.ndarray] = None, barrierIsobarCompensation: Callable[[int, np.ndarray], np.ndarray] = None) -> None:
		"""Add a wave to the PWA Model

		Args:
			waveName (str): Name of the wave
			hadronicTensorName (str): Label of the hadronic tensor
			dynamicXAmplitude (Callable[[np.ndarray], np.ndarray]): Dynamic amplitude of the X system as function of the :math:`3\\pi` mass
			dynamicIsobarAmplitude (Callable[[np.ndarray], np.ndarray]): Dynamic amplitude of the X system as function of the :math:`2\\pi` mass
			barrierXCompensation (Callable[[int, np.ndarray, np.ndarray], np.ndarray], optional): Angular momentum barrier compensation factor in the X decay. Defaults to None.
			                                                                          This is a function of (L, m123, m12). L will be parsed from the hadronicTensorName.
			barrierIsobarCompensation (Callable[[int, np.ndarray], np.ndarray], optional): Angular momentum barrier compensation factor in the X decay. Defaults to None.
			                                                                               This is a function of (L, m12). L will be parsed from the hadronicTensorName.
		"""

		if waveName in self._waveNames:
			log.raiseException(ArgumentError, f'Wave "{waveName}" already in model!')
		self._waveNames.append(waveName)

		self._waveIdx2HadronicTensor.append(hadronicTensorName)

		if dynamicIsobarAmplitude not in self._dynamicIsobarAmplitudes:
			self._dynamicIsobarAmplitudes.append(dynamicIsobarAmplitude)
		self._waveIdx2DynamicIsobarAmplitudeIdx.append(self._dynamicIsobarAmplitudes.index(dynamicIsobarAmplitude))

		if barrierIsobarCompensation is None:
			def barrierIsobarComp(M: Any) -> float: # pylint: disable=unused-argument
				return 1.0
		else:
			barrierIsobarComp = amplitudes.LambdaF(barrierIsobarCompensation, L=Wave.fromString(hadronicTensorName).J_isobar)
		if barrierIsobarComp not in self._barrierIsobarCompensation:
			self._barrierIsobarCompensation.append(barrierIsobarComp)
		self._waveIdx2BarrierIsobarCompensationIdx.append(self._barrierIsobarCompensation.index(barrierIsobarComp))

		if dynamicXAmplitude not in self._dynamicXAmplitudes:
			self._dynamicXAmplitudes.append(dynamicXAmplitude)
		self._waveIdx2DynamicXAmplitudeIdx.append(self._dynamicXAmplitudes.index(dynamicXAmplitude))

		if barrierXCompensation is None:
			def barrierXComp(M: Any, m1: Any) -> float: # pylint: disable=unused-argument
				return 1.0
		else:
			barrierXComp = amplitudes.LambdaF(barrierXCompensation, L=Wave.fromString(hadronicTensorName).L)
		if barrierXComp not in self._barrierXCompensation:
			self._barrierXCompensation.append(barrierXComp)
		self._waveIdx2BarrierXCompensationIdx.append(self._barrierXCompensation.index(barrierXComp))



	def setDecayAmplitudesIntegratedNormIntegrals(self, integralsForParticle: np.ndarray, integralsForAntiparticle: np.ndarray) -> None:
		'''Set normalization integrals for decay amplitudes for particles or antiparticles.

		:param integralsForParticle: Array of the shape (nWaves, 1, 1) containing the normalization integrals for each wave for the particle decay amplitudes
		:type integralsForParticle: np.ndarray
		:param integralsForAntiparticle: Array of the shape (nWaves, 1, 1) containing the normalization integrals for each wave for the antiparticle decay amplitudes
		:type integralsForAntiparticle: np.ndarray
		'''
		if integralsForParticle.shape != (self.nWaves, 1, 1):
			raise Exception("The shape of the integrals must be (nWaves, 1, 1)!")
		if integralsForAntiparticle.shape != (self.nWaves, 1, 1):
			raise Exception("The shape of the integrals must be (nWaves, 1, 1)!")
		self._decayAmplitudesIntegratedNormIntegralsForParticles = integralsForParticle
		self._decayAmplitudesIntegratedNormIntegralsForAntiparticles = integralsForAntiparticle


	def calcTotalHadronicCurrent(self, events: EventsTau2ThreeChargedPi, partialWaveAmplitudes: dict, normalized: bool = True) -> np.ndarray:
		'''Return total hadronic current from all partial wave contributions. It is normalized if `normalized` is set to `True` and normalization integrals are given.

		There is a :math:`2\\pi` factor between the normalization integral of the unpolarized decay amplitude and the integrated unpolarized decay amplitude.

		:param events: Events of the :math:`\\tau \\to 3 \\pi` decay
		:type events: EventsTau2ThreeChargedPi
		:param partialWaveAmplitudes: Dictionary of partial waves and their complex coefficients
		:type partialWaveAmplitudes: dict
		:param normalized: Apply normalization if set to True, defaults to True
		:type normalized: bool, optional
		:return: Total hadronic current
		:rtype: np.ndarray
		'''
		currents = self.calcHadronicCurrents(events)
		totalCurrent = math.full_like(currents[0], 0.)
		for waveIdx, waveName in enumerate(self.waveNames):
			waveCurrent = partialWaveAmplitudes[waveName] * currents[waveIdx]
			if normalized:
				if self._decayAmplitudesIntegratedNormIntegralsForParticles is None:
					raise Exception("Normalization integrals not given!")
				# there is a 2*pi factor between the normalization integral of the unpolarized decay amplitude
				# and the integrated unpolarized decay amplitude
				waveCurrent[...,events.isParticle] /= math.sqrt(self._decayAmplitudesIntegratedNormIntegralsForParticles[waveIdx]/2./np.pi)
				waveCurrent[...,~events.isParticle] /= math.sqrt(self._decayAmplitudesIntegratedNormIntegralsForAntiparticles[waveIdx]/2./np.pi)
			totalCurrent += waveCurrent
		return totalCurrent


	def calcHadronicCurrents(self, events: EventsTau2ThreeChargedPi) -> list:
		'''Return list of symmetrized hadronic currents for all partial waves.

		:param events: Kinematics of events from the :math:`\\tau \\to 3 \\pi + \\nu_{\\tau}`
		:type events: EventsTau2ThreeChargedPi
		:return: Hadronic currents of partial waves
		:rtype: list
		'''
		#                                                   --------- isobar ---------
		currents12 = self.calcHadronicCurrentsUnsymmetrized(events.pi1PM, events.pi2MP, events.pi3MP)
		currents13 = self.calcHadronicCurrentsUnsymmetrized(events.pi1PM, events.pi3MP, events.pi2MP)
		return [current12 + current13 for current12, current13 in zip(currents12, currents13)]


	def calcHadronicCurrentsUnsymmetrized(self, pi1PM: np.ndarray, pi2MP: np.ndarray, pi3MP: np.ndarray) ->  list:
		'''
		Calculate hadronic currents. The isobar system is the (12) system, i.e. :math:`\\pi^{\\pm}_1 \\pi^{\\mp}_2`
		'''
		self._hadronicTensors.setEvents(pi1PM, pi2MP, pi3MP )

		dynamicIsobarAmplitudeValues = []
		for dynamicAmplitude in self._dynamicIsobarAmplitudes:
			dynamicIsobarAmplitudeValues.append(dynamicAmplitude(self._hadronicTensors.m12))

		barrierIsobarCompensationValues = []
		for barrierCompensation in self._barrierIsobarCompensation:
			barrierIsobarCompensationValues.append(barrierCompensation(M=self._hadronicTensors.m12))

		dynamicXAmplitudeValues = []
		for dynamicAmplitude in self._dynamicXAmplitudes:
			dynamicXAmplitudeValues.append(dynamicAmplitude(self._hadronicTensors.m123))

		barrierXCompensationValues = []
		for barrierCompensation in self._barrierXCompensation:
			barrierXCompensationValues.append(barrierCompensation(M=self._hadronicTensors.m123, m1=self._hadronicTensors.m12))

		currents = []
		for waveIdx in range(self.nWaves):
			tensor = self._hadronicTensors[self._waveIdx2HadronicTensor[waveIdx]]
			current = tensor \
			         * dynamicIsobarAmplitudeValues[self._waveIdx2DynamicIsobarAmplitudeIdx[waveIdx]] \
			         * barrierIsobarCompensationValues[self._waveIdx2BarrierIsobarCompensationIdx[waveIdx]] \
			         * dynamicXAmplitudeValues[self._waveIdx2DynamicXAmplitudeIdx[waveIdx]] \
			         * barrierXCompensationValues[self._waveIdx2BarrierXCompensationIdx[waveIdx]]

			currents.append(current)
		return currents


	def calcLeptonicCurrent(self, tau: np.ndarray, nu: np.ndarray, upTau: bool, isParticle: np.ndarray) -> np.ndarray:
		'''
		Calculate the leptonic current for the given event

		:param tau: 4-momentum of the tau
		:param nu: 4-momentum of the neutrino
		:param upTau: Helicity of the :math:`\\tau` is up, the helicity of the neutrino is always down for :math:`\\tau^-` decays and up for :math:`\\tau^+` decays
		:param isParticle: Bool array, which is True for :math:`\\tau^-` decause and false for :math:`\\tau^+` decays
		'''
		leptonicCurrent = math.empty(tau.shape, tau, dtype=np.complex128)
		leptonicCurrent[:,isParticle] = self._dirac.weakLepCurr_particledecay(tau[:,isParticle], nu[:,isParticle], upTau=upTau, upNu=False)
		leptonicCurrent[:,~isParticle] = self._dirac.weakLepCurr_antiparticledecay(tau[:,~isParticle], nu[:,~isParticle], upTau=upTau, upNu=True)
		return leptonicCurrent


	def calcLeptonicTensorUnpolarized(self, tau: np.ndarray, nu:np.ndarray, isParticle: np.ndarray) -> np.ndarray:
		'''
		Calculate the unpolarized leptonic tensor

		:param tau: 4-momentum of the tau
		:param tau: 4-momentum of the neutrino
		'''
		return self._dirac.weakLepTensor_unpolarizedDecay(tau, nu, isParticle)


	def calcLeptonicTensorIntegratedUnpolarized(self, events: EventsTau2ThreeChargedPi):
		'''
		Calculate the unpolarized leptonic current, integrated over the unknown angle :math:`\\alpha` of the :math:`\\tau` momentum around the measured hadronic momentum

		:param events: The kinematics MUST be given in the :math:`e^+ e^-` CMS, the tau and neutrino 4-momenta from `events` are NOT used
		:param tauE: Energy of the :math:`\\tau` in the :math:`e^+e^-` CMS
		:param tauM: Rest mass of the :math:`\\tau`
		'''

		p123 = events.pi1PM + events.pi2MP + events.pi3MP
		pTauParallel, pTauOrtho1, pTauOrtho2 = getTauMomentumComponentsCMS(p123, events.tauMP_E, events.tauMP_M)


		leptonicTensor = 2.*np.pi*self.calcLeptonicTensorUnpolarized(tau=pTauParallel, nu=pTauParallel-p123, isParticle=events.isParticle)
		leptonicTensor +=   np.pi*self.calcLeptonicTensorUnpolarized(tau=pTauOrtho1,   nu=pTauOrtho1,        isParticle=events.isParticle)
		leptonicTensor +=   np.pi*self.calcLeptonicTensorUnpolarized(tau=pTauOrtho2,   nu=pTauOrtho2,        isParticle=events.isParticle)

		return leptonicTensor

	def calcIntensityUnpolarized(self, events: EventsTau2ThreeChargedPi, partialWaveAmplitudes: dict, normalized: bool = True):
		'''
		Calculate the total model intensity for the given partial-wave amplitudes by averaging over the helicities of the :math:`\\tau`.
		The given :math:`\\tau` and :math:`\\nu` 4-momenta from the `events` are used.

		:param events: Kinematics of events from the :math:`\\tau \\to 3 \\pi + \\nu_{\\tau}` decay
		:type events: EventsTau2ThreeChargedPi
		:param partialWaveAmplitudes: Dictionary of partial waves and their complex coefficients
		:type partialWaveAmplitudes: dict
		:param normalized: Apply normalization if set to True, defaults to True
		:type normalized: bool, optional
		:return: Total intensity of given partial wave amplitudes
		'''
		totalHadronicCurrent = self.calcTotalHadronicCurrent(events, partialWaveAmplitudes, normalized=normalized)
		intensityUp   = math.abs2( lorentz.lp(totalHadronicCurrent, self.calcLeptonicCurrent(events.tauMP, events.nu, True,  events.isParticle)) )
		intensityDown = math.abs2( lorentz.lp(totalHadronicCurrent, self.calcLeptonicCurrent(events.tauMP, events.nu, False, events.isParticle)) )
		return 0.5*( intensityUp + intensityDown )


	def calcDecayAmplitudesIntegrated(self, events: EventsTau2ThreeChargedPi, normalized: bool=True):
		'''
		Calculate the decay amplitudes for the decay of an unpolarized :math:`\\tau`, integrated over the unknown angle :math:`\\alpha` of the :math:`\\tau` momentum
		around the measured hadronic momentum.
		The output array has dimensions (nWaves, nLeptonicVecotors, nEvents), where nLeptonicVectors=4 and arises from the decomposition of the leptonic tensor.

		:param events: The kinematics MUST be given in the :math:`e^+ e^-` CMS, the tau and neutrino 4-momenta from `events` are NOT used
		:param normalized: Normalize the decay amplitudes.
		'''

		hadronicCurrents = self.calcHadronicCurrents(events)
		leptonicTensor = self.calcLeptonicTensorIntegratedUnpolarized(events)
		leptonicVectors =  decomposeHermitianMatrix2(leptonicTensor)

		# quick check
		if np.max(np.abs( leptonicTensor - math.einsum('ime,ine->mne', leptonicVectors, math.conjugate(leptonicVectors)) )) > 1e-10:
			raise Exception("Vector decomposition of leptonic tensor did not work!")


		decayAmplitudes = np.empty((self.nWaves, 4, events.nEvents), dtype=np.complex128)
		for waveIdx in range(self.nWaves):
			for leptonicVecorIdx in range(4):
				decayAmplitudes[waveIdx][leptonicVecorIdx] = lorentz.lp(hadronicCurrents[waveIdx], leptonicVectors[leptonicVecorIdx])

		if normalized:
			if self._decayAmplitudesIntegratedNormIntegralsForParticles is None:
				raise Exception("Normalization integrals not given!")
			decayAmplitudes[...,events.isParticle] /= math.sqrt(self._decayAmplitudesIntegratedNormIntegralsForParticles)
			decayAmplitudes[...,~events.isParticle] /= math.sqrt(self._decayAmplitudesIntegratedNormIntegralsForAntiparticles)

		# checkAmplitudes
		if np.any(np.isnan(decayAmplitudes)):
			raise Exception("Some of the decay amplitudes is NaN! You ma want to use `np.seterr('raise')`.")
		return decayAmplitudes


	def calcDecayAmplitudes(self, events: EventsTau2ThreeChargedPi, normalized: bool=True):
		'''
		Calculate the decay amplitudes for the decay of an unpolarized tau.
		The output array has dimensions (nWaves, nLeptonicVecotors, nEvents), where nLeptonicVectors=4 and arises from the decomposition of the leptonic tensor.

		:param events: Kinematics of events from the :math:`\\tau \\to 3 \\pi + \\nu_{\\tau}` decay
		:param normalized: Normalize the decay amplitudes.
		'''
		if events.tauMP.size == 0:
			raise Exception("Tau momenta not stored in events!")
		if events.nu.size == 0:
			raise Exception("Tau momenta not stored in events!")
		hadronicCurrents = self.calcHadronicCurrents(events)
		leptonicTensor = self.calcLeptonicTensorUnpolarized(events.tauMP, events.nu, events.isParticle)
		leptonicVectors = decomposeHermitianMatrix2(leptonicTensor)
		decayAmplitudes = np.empty((self.nWaves, 4, events.nEvents), dtype=np.complex128)
		for waveIdx in range(self.nWaves):
			for leptonicVecorIdx in range(4):
				decayAmplitudes[waveIdx][leptonicVecorIdx] = lorentz.lp(hadronicCurrents[waveIdx], leptonicVectors[leptonicVecorIdx])

		if normalized:
			if self._decayAmplitudesIntegratedNormIntegralsForParticles is None:
				raise Exception("Normalization integrals not given!")
			# there is a 2*pi factor between the normalization integral of the unpolarized decay amplitude
			# and the integrated unpolarized decay amplitude
			decayAmplitudes[...,events.isParticle] /= math.sqrt(self._decayAmplitudesIntegratedNormIntegralsForParticles/2./np.pi)
			decayAmplitudes[...,~events.isParticle] /= math.sqrt(self._decayAmplitudesIntegratedNormIntegralsForAntiparticles/2./np.pi)

		# checkAmplitudes
		if np.any(np.isnan(decayAmplitudes)):
			raise Exception("Some of the decay amplitudes is NaN! You ma want to use `np.seterr('raise')`.")

		return decayAmplitudes
