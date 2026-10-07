# coding: utf-8
'''
Created on Tuesday 29 03 2022
@author: Stefan Wallner
@description: Casses for start-parameter generators
'''

from __future__ import absolute_import, print_function, division

import numpy as np

from ... import math
from ... import random


class StartparameterGenerator():
	def __init__(self,nWaves: int, iReslPosAmps: np.ndarray ) -> None:
		'''
		@param nWaves: Number of partial waves, i.e. number of coupling amplitudes
		@param iRealAmps: Number of real-valued positive amplitudes, e.g. ref. waves
		'''
		self._nWaves = nWaves
		self._iRealPosAmps = iReslPosAmps

	def generateCouplings(self) -> np.ndarray:
		'''
		Generate a set of random start-parameter values for the coupling amplitudes.
		'''
		raise NotImplementedError()


class UniformStartparameterGenerator(StartparameterGenerator):
	'''
	Generate real and imaginary parts flat in the range -sqrt(nEvents), sqrt(nEvents)
	'''
	def __init__(self, nWaves: int, iRealAmps: np.ndarray, nEvents: int, nBkg: int, nBkgEvents: int) -> None:
		super().__init__(nWaves, iRealAmps)
		self._nEvents = nEvents
		self._nBkg = nBkg
		self._nBkgEvents = nBkgEvents

	def generateCouplings(self) -> np.ndarray:
		'''
		Generate a set of random start-parameter values for the coupling amplitudes.
		'''
		rndm = random.generators.pwaStartparameter.uniform(-math.sqrt(self._nEvents), math.sqrt(self._nEvents), self._nWaves*2)
		couplings = rndm[:self._nWaves] + 1j*rndm[self._nWaves:]
		couplings[self._iRealPosAmps] = math.abs(math.real(couplings[self._iRealPosAmps]))
		return couplings

	def generateBackgroundYields(self) -> np.ndarray:
		return random.generators.pwaStartparameter.uniform(1, self._nBkgEvents, self._nBkg)

	def generateIntensitiesandPhases(self) -> np.ndarray:
		intensities = random.generators.pwaStartparameter.uniform(0.0, 0.1*(3*self._nEvents), self._nWaves)
		intensities[self._iRealPosAmps] = random.generators.pwaStartparameter.uniform(0.9*(3*self._nEvents), 1.1*(3*self._nEvents), np.size(self._iRealPosAmps))

		phases = random.generators.pwaStartparameter.uniform(0, 2* np.pi , self._nWaves)
		phases[self._iRealPosAmps] = 0.0

		return intensities, phases
