# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Skeleton model for tau -> two-body decays
'''

from __future__ import absolute_import, print_function, division

from argparse import ArgumentError
from typing import Callable, Any, Optional

import numpy as np

from .... import dirac
from .... import math
from .... import lorentz
from .... import amplitudes
from ....utils import Logger
from ..._model import PWAModel
from .._hadronicTensor import HadronicTensor, Wave



log = Logger("tau2")


class Tau2Twobody(PWAModel):
	'''
	PWA model for :math:`\\tau \\to [(1) (2)] + \\nu_\\tau` decays.
	'''

	def __init__(self, name: str, description: Optional[str] = None) -> None:
		super().__init__(name, description if description is not None else "")
		self._hadronicTensor = HadronicTensor()
		self._dirac = dirac.Weyl()
		self._waveIdx2HadronicTensor = []
		self._dynamicXAmplitudes = []
		self._waveIdx2DynamicXAmplitudeIdx = []
		self._barrierXCompensation = []
		self._waveIdx2BarrierXCompensationIdx = []
		self._decayAmplitudesIntegratedNormIntegralsForParticles = None  # normalization integrals for integrated decay amplitudes: shape (1, nWAves, 1)
		self._decayAmplitudesIntegratedNormIntegralsForAntiparticles = None  # normalization integrals for integrated decay amplitudes: shape (1, nWAves, 1)


	def addWave(self,
	            waveName: str,
	            hadronicTensorName: str,
	            dynamicXAmplitude: Callable[[np.ndarray], np.ndarray],
	            barrierXCompensation: Optional[Callable[[Any], Any]] = None) -> None:
		'''
		Add a wave definition to the model.
		'''
		if waveName in self._waveNames:
			log.raiseException(ArgumentError, f'Wave "{waveName}" already in model!')
		self._waveNames.append(waveName)
		self._waveIdx2HadronicTensor.append(hadronicTensorName)

		if dynamicXAmplitude not in self._dynamicXAmplitudes:
			self._dynamicXAmplitudes.append(dynamicXAmplitude)
		self._waveIdx2DynamicXAmplitudeIdx.append(self._dynamicXAmplitudes.index(dynamicXAmplitude))

		if barrierXCompensation is None:
			def barrierXComp(M: Any):  # pylint: disable=unused-argument
				return 1.0
		else:
			barrierXComp = amplitudes.LambdaF(barrierXCompensation, L=Wave.fromString(hadronicTensorName).J_X)
		if barrierXComp not in self._barrierXCompensation:
			self._barrierXCompensation.append(barrierXComp)
		self._waveIdx2BarrierXCompensationIdx.append(self._barrierXCompensation.index(barrierXComp))


	def calcTotalHadronicCurrent(self,
	                            p1: np.ndarray,
	                            p2: np.ndarray,
	                            partialWaveAmplitudes: dict,
	                            normalized: bool = True) -> np.ndarray:
		'''Return total hadronic current from all partial-wave contributions.

		If `normalized` is True, wave currents are scaled by the stored integrated normalization
		integrals and the `isParticle` mask to separate particle and antiparticle events.
		'''
		currents = self.calcHadronicCurrents(p1, p2)
		totalCurrent = math.full_like(currents[0], 0.)
		for waveIdx, waveName in enumerate(self.waveNames):
			waveCurrent = partialWaveAmplitudes[waveName] * currents[waveIdx]
			if normalized:
				if self._decayAmplitudesIntegratedNormIntegralsForParticles is None or self._decayAmplitudesIntegratedNormIntegralsForAntiparticles is None:
					raise Exception('Normalization integrals not given!')
				raise Exception('Normalization of decay amplitudes not implemented yet!')
			totalCurrent += waveCurrent
		return totalCurrent


	def calcHadronicCurrents(self, p1: np.ndarray, p2: np.ndarray) -> list:
		'''
		Evaluate hadronic-current building blocks for all configured waves.
		'''
		self._hadronicTensor.setEvents(p1, p2)

		dynamicXAmplitudeValues = []
		for dynamicAmplitude in self._dynamicXAmplitudes:
			dynamicXAmplitudeValues.append(dynamicAmplitude(self._hadronicTensor.m12))

		barrierXCompensationValues = []
		for barrierCompensation in self._barrierXCompensation:
			barrierXCompensationValues.append(barrierCompensation(M=self._hadronicTensor.m12))

		currents = []
		for waveIdx in range(self.nWaves):
			tensor = self._hadronicTensor[self._waveIdx2HadronicTensor[waveIdx]]
			current = tensor \
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
		:param isParticle: Bool array, which is True for :math:`\\tau^-` decays and false for :math:`\\tau^+` decays
		'''
		leptonicCurrent = math.empty(tau.shape, tau, dtype=np.complex128)
		leptonicCurrent[:,isParticle] = self._dirac.weakLepCurr_particledecay(tau[:,isParticle], nu[:,isParticle], upTau=upTau, upNu=False)
		leptonicCurrent[:,~isParticle] = self._dirac.weakLepCurr_antiparticledecay(tau[:,~isParticle], nu[:,~isParticle], upTau=upTau, upNu=True)
		return leptonicCurrent


	def calcIntensityUnpolarized(self,
	                            p1: np.ndarray,
	                            p2: np.ndarray,
	                            tau: np.ndarray,
	                            nu: np.ndarray,
	                            isParticle: np.ndarray,
	                            partialWaveAmplitudes: dict,
	                            normalized: bool = True) -> np.ndarray:
		'''Calculate the total model intensity by averaging over tau helicities.

		The intensity is computed as the average of contributions from tau spin-up and spin-down:
		I = 0.5 * (I_up + I_down)

		:param p1: 4-momentum of particle 1
		:param p2: 4-momentum of particle 2
		:param tau: 4-momentum of the tau
		:param nu: 4-momentum of the neutrino
		:param isParticle: Bool array indicating particle (True) or antiparticle (False) events
		:param partialWaveAmplitudes: Dictionary of partial waves and their complex coefficients
		:param normalized: Apply normalization if True
		:return: Total intensity for each event
		'''
		totalHadronicCurrent = self.calcTotalHadronicCurrent(p1, p2, partialWaveAmplitudes, normalized=normalized)
		intensityUp = math.abs2(lorentz.lp(totalHadronicCurrent, self.calcLeptonicCurrent(tau, nu, True, isParticle)))
		intensityDown = math.abs2(lorentz.lp(totalHadronicCurrent, self.calcLeptonicCurrent(tau, nu, False, isParticle)))
		return 0.5 * (intensityUp + intensityDown)




	def calcIntensityPolarized(self,
	                            p1: np.ndarray,
	                            p2: np.ndarray,
	                            tau: np.ndarray,
	                            nu: np.ndarray,
	                            isParticle: np.ndarray,
	                            partialWaveAmplitudes: dict,
								polarization: float,
	                            normalized: bool = True) -> np.ndarray:
		'''Calculate the total model intensity by averaging over tau helicities.

		The intensity is computed as the average of contributions from tau spin-up and spin-down:
		I = 0.5 * (I_up + I_down)

		:param p1: 4-momentum of particle 1
		:param p2: 4-momentum of particle 2
		:param tau: 4-momentum of the tau
		:param nu: 4-momentum of the neutrino
		:param isParticle: Bool array indicating particle (True) or antiparticle (False) events
		:param partialWaveAmplitudes: Dictionary of partial waves and their complex coefficients
		:param polarization: Polarization parameter in the range [-1, 1], where -1 corresponds to fully polarized tau with spin down,
		                     +1 to fully polarized tau with spin up, and 0 to unpolarized tau
		:param normalized: Apply normalization if True
		:return: Total intensity for each event
		'''
		totalHadronicCurrent = self.calcTotalHadronicCurrent(p1, p2, partialWaveAmplitudes, normalized=normalized)
		intensityUp = math.abs2(lorentz.lp(totalHadronicCurrent, self.calcLeptonicCurrent(tau, nu, True, isParticle)))
		intensityDown = math.abs2(lorentz.lp(totalHadronicCurrent, self.calcLeptonicCurrent(tau, nu, False, isParticle)))
		return 0.5 * ((1 + polarization) * intensityUp + (1 - polarization) * intensityDown)
