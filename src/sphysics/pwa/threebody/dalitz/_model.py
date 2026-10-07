# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Model for Dalitz Plot fits, Created on Monday 30 01 2023
'''

from __future__ import absolute_import, print_function, division, annotations

from typing import Callable
import numpy as np
import tensorflow as tf

from .... import math
from ...._constants import Constants as C
from ....utils import Logger
from ... import barrierFactors
from ..._model import PWAModel
from .._dalitzAmplitudes import DalitzAmplitudes3P_Zemach


FIsobarDecay = barrierFactors.compensationFactor_BlattWeisskopf # pylint: disable=invalid-name
FBDecay = barrierFactors.compensationFactor_DalitzBlattWeisskopf # pylint: disable=invalid-name

log = Logger("dalitz")


class PTo3P(PWAModel):
	'''
	PWA model for :math:`P \\to P_1 P_2 P_3`.

	Model for the decay of a pseudoscalar particle to three pseudoscalar particles
	The order of the particles is as given by the above reaction.
	The spin analyzer, i.e. the direction of the particle that defines :math:`\\cos(\\theta_H)` is:

	- the :math:`P_2` for the :math:`P_1 P_2` isobar
	- the :math:`P_3` for the :math:`P_1 P_3` isobar
	- the :math:`P_3` for the :math:`P_2 P_3` isobar
	'''

	m123 = None
	m1   = None
	m2   = None
	m3   = None
	isobarSystemNames = {} # Map from isobar system names to particle indices, e.g. {'Kpi': '12'} for K pi pi final state

	def __init__(self,
				 name: str,
				 description: str = None,
				 qR_P: float = C.barrierFactorMomentumscale) -> None:
		"""Construct a new model

		Args:
			name (str): Model name
			isobarSystemNames (dict):
			description (str, optional): Model description. Defaults to None.
			qR_P (float, optional): Momentum scale (1/radius) for angular momentum barrier factor of P-> isobar + P decay. Defaults to C.barrierFactorMomentumscale.
		"""
		super().__init__(name, description)
		self._dalitzAmplitudes = {
			'12': DalitzAmplitudes3P_Zemach(),
			'13': DalitzAmplitudes3P_Zemach(),
			'23': DalitzAmplitudes3P_Zemach()
		}

		self._waveIdx2AmplitudeName = []

		self._dynamicIsobarAmplitudes = []
		self._isobarSystem = []
		self._waveIdx2DynamicIsobarAmplitudeIdx = []
		self._decayAmplitudesNormIntegrals = None  # normalization integrals for integrated decay amplitudes: shape (1, nWAves, 1)
		self._jIsobar = []
		self._m0Isobar = []
		self._qR = []
		self._q0 = qR_P

	def setDecayAmplitudesNormIntegrals(self, normIntegrals: np.ndarray):
		"""Set normalization integrals for decay amplitudes

		Args:
			normIntegrals (np.ndarray): Normalization integral for each partial wave of the shape [<waveindex>]
		"""
		self._decayAmplitudesNormIntegrals = normIntegrals

	def addWave(self, waveName: str, hadronicTensorName: str,
				dynamicIsobarAmplitude: Callable, isobarSystem: str,
				jIsobar: int, m0Isobar: float, qR: float):
		"""Add a further partial wave to the model

		Args:
			waveName (str): Name of the wave
			hadronicTensorName (str): Name of the hadronic tensor in the DalitzAmplitudes3p_Zemach formalism
			dynamicIsobarAmplitude (Callable): Isobar-mass dependence of the isobar dynamci amplitude, e.g. Breit Wigner amplitude
			isobarSystem (str): Name of the isobar system is given in `isobarSystemNames`
			Jisobar (int): Spin of the isobar
			m0Isobar (float): Nominal mass of the isobar for the normalization of the angular momentum barrier factors
			qR (float): Momentum scaling factor (1/radius) for the angular momentum barrier factor of the isobar decay. If `None`, no barrier factor is used
		"""
		if isobarSystem in self.isobarSystemNames:
			isobarSystem = self.isobarSystemNames[isobarSystem]
		else:
			log.raiseException(ValueError,
							   f'Unknown isobarSystem {isobarSystem}!')

		if waveName in self._waveNames:
			log.raiseException(ValueError,
							   f'Wave "{waveName}" already in model!')
		self._waveNames.append(waveName)

		self._waveIdx2AmplitudeName.append(hadronicTensorName)

		if dynamicIsobarAmplitude not in self._dynamicIsobarAmplitudes:
			self._dynamicIsobarAmplitudes.append(dynamicIsobarAmplitude)
			self._isobarSystem.append(isobarSystem)
		if self._isobarSystem[self._dynamicIsobarAmplitudes.index(
			dynamicIsobarAmplitude)] != isobarSystem:
			log.raiseException(Exception, "Different isobar system")
		self._waveIdx2DynamicIsobarAmplitudeIdx.append(
			self._dynamicIsobarAmplitudes.index(dynamicIsobarAmplitude))

		self._jIsobar.append(jIsobar)
		self._qR.append(qR)
		self._m0Isobar.append(m0Isobar)

	def calcDecayAmplitudes(self,
							p1: np.ndarray|tf.Tensor,
							p2: np.ndarray|tf.Tensor,
							p3: np.ndarray|tf.Tensor,
							normalized: bool = True,
							batchSize: int = 100_000) -> np.ndarray|tf.Tensor:
		"""Calculate the decay amplitudes

		Args:
			p1 (np.ndarray): 4-momentum of the 1. final-state particle of shape [<4>,<nEvents>]
			p2 (np.ndarray): 4-momentum of the 2. final-state particle of shape [<4>,<nEvents>]
			p3 (np.ndarray): 4-momentum of the 3. final-state particle of shape [<4>,<nEvents>]
			normalized (bool, optional): Normalize the decay amplitudes by the corresponding normalization
										 integral set via `setDecayAmplitudesNormIntegrals`. Defaults to True.
			batchSize (int, optional): Decay amplitudes are calculated in event batches of `batchSize` to stay in the cache.
									   Defaults to 100_000.

		Returns:
			np.ndarray: Array with decay amplitudes of shape [<nWaves>,<nEvents>]
		"""
		decayAmplitudes = None
		if batchSize is not None:
			decayAmplitudes = math.empty((self.nWaves, p1.shape[1]),
									p1, dtype=np.complex128)
			batchSize = min(batchSize, p1.shape[1])
			nBatches = int(p1.shape[1] // batchSize)
			q = p1.shape[1] // nBatches
			m = p1.shape[1] % nBatches
			for iBatch in range(nBatches):
				iStart = iBatch * q + min(m, iBatch)
				iEnd = (iBatch + 1) * q + min(m, iBatch + 1)
				decayAmplitudes[:, iStart:iEnd] = self.calcDecayAmplitudes(
					p1[:, iStart:iEnd],
					p2[:, iStart:iEnd],
					p3[:, iStart:iEnd],
					normalized=normalized,
					batchSize=None)
		else:
			self._dalitzAmplitudes['12'].setEvents(p2, p1, p3)  # the charged pion is the spin analyzer pylint: disable=arguments-out-of-order
			self._dalitzAmplitudes['13'].setEvents(p1, p3, p2)  # the charged kaon is the spin analyzer pylint: disable=arguments-out-of-order
			self._dalitzAmplitudes['23'].setEvents(p3, p2, p1)  # the pi0 is the spin analyzer pylint: disable=arguments-out-of-order
			decayAmplitudes = self._calcDecayAmplitudes(normalized=normalized)
		return decayAmplitudes


	def calcDecayAmplitudesFromMasses(self,
	                                  m12: np.ndarray|tf.Tensor, m23: np.ndarray|tf.Tensor,
								      normalized: bool = True) -> np.ndarray|tf.Tensor:
		"""Calculate decay amplitudes from invariant masses

		Args:
			m12 (np.ndarray): Invariant mass of the (12) system
			m23 (np.ndarray): Invariant mass of the (23) system
			normalized (bool, optional): Normalize the decay amplitudes by the corresponding normalization
										 integral set via `setDecayAmplitudesNormIntegrals`. Defaults to True.

		Returns:
			np.ndarray | tf.Tensor: Array with decay amplitudes of shape [<nWaves>,<nEvents>]
		"""
		m123 = self.m123
		m1   = self.m1
		m2   = self.m2
		m3   = self.m3
		m13 = math.sqrt(m123**2 + m1**2 + m2**2 + m3**2 - m12**2 - m23**2)

		# p1p2  : p2 --> (12) & (23), p2 is the spin analyzer
		self._dalitzAmplitudes['12'].setMasses(m123=m123, m12=m12, m13=m23, m23=m13, m1=m2, m2=m1, m3=m3)

		# p1p3  : p1  --> (13) & (12), p1 is the spin analyzer
		self._dalitzAmplitudes['13'].setMasses(m123=m123, m12=m13, m13=m12, m23=m23, m1=m1, m2=m3, m3=m2)

		# p2p3 : p3 --> (23) & (13), p3 is the spin analyzer
		self._dalitzAmplitudes['23'].setMasses(m123=m123, m12=m23, m13=m13, m23=m12, m1=m3, m2=m2, m3=m1)

		return self._calcDecayAmplitudes(normalized=normalized)



	def _calcDecayAmplitudes(self, normalized: bool) -> np.ndarray | tf.Tensor:
		"""Calculate decay amplitudes if kinematics of events was set for all `self._dalitsAmplitudes`

		Returns:
			np.ndarray | tf.Tensor: Array with decay amplitudes of shape [<nWaves>,<nEvents>]
		"""
		decayAmplitudes = []
		dynamicIsobarAmplitudeValues = []
		for dynamicAmplitude, isobarSystem in zip(
			self._dynamicIsobarAmplitudes, self._isobarSystem):
			dynamicIsobarAmplitudeValues.append(
				dynamicAmplitude(self._dalitzAmplitudes[isobarSystem].m12))

		for waveIdx in range(self.nWaves):
			isobarSystem = self._isobarSystem[self._waveIdx2DynamicIsobarAmplitudeIdx[waveIdx]]
			dalitzAmpl = self._dalitzAmplitudes[isobarSystem]

			dalitzAmpl_ = math.castToComplex(dalitzAmpl[self._waveIdx2AmplitudeName[waveIdx]])
			dynamicAmpl_ = math.castToComplex(dynamicIsobarAmplitudeValues[self._waveIdx2DynamicIsobarAmplitudeIdx[waveIdx]])

			f_B = math.castToComplex(FBDecay(self._jIsobar[waveIdx], dalitzAmpl.m12**2, dalitzAmpl.m123,  dalitzAmpl.m3, self._m0Isobar[waveIdx], self._q0))

			f_res = math.castToComplex((FIsobarDecay(self._jIsobar[waveIdx], dalitzAmpl.m12**2,  dalitzAmpl.m1,  dalitzAmpl.m2, self._m0Isobar[waveIdx],
							   self._qR[waveIdx]) if self._qR[waveIdx] is not None else 1.0))

			amplitude = dalitzAmpl_ * dynamicAmpl_ * f_B * f_res

			decayAmplitudes.append(amplitude)
		decayAmplitudes = math.stack(decayAmplitudes)

		if normalized:
			decayAmplitudes /= math.sqrt(
				self._decayAmplitudesNormIntegrals[:, None])

		return decayAmplitudes



class B0ToKpipi0(PTo3P):
	"""
    PWA model for :math:`B_0 \\to K^+ \\pi^- \\pi^0`

    The order of the particles is as given by the above reaction.

    The spin analyzer, i.e. the direction of the particle that defines :math:`\\cos(\\theta_H)` is:

    - the :math:`\\pi^-` for the :math:`K^+ \\pi^-` isobar (isobarSystemName = 'Kst0')
    - the :math:`K^+` for the :math:`K^+ \\pi^0` isobar (isobarSystemName = 'KstP')
    - the :math:`\\pi^0` for the :math:`\\pi^- \\pi^0` isobar (isobarSystemName = 'Rho')

    Args:
        p1 (np.ndarray): 4-momentum of the :math:`K^+` final-state particle, shape [<4>,<nEvents>].
        p2 (np.ndarray): 4-momentum of the :math:`\\pi^-` final-state particle, shape [<4>,<nEvents>].
        p3 (np.ndarray): 4-momentum of the :math:`\\pi^0` final-state particle, shape [<4>,<nEvents>].
    """
	m123 = C.M.B0
	m1 = C.M.K
	m2 = C.M.pi
	m3 = C.M.pi0
	isobarSystemNames = {'Kst0': '12', 'KstP': '13', 'Rho': '23'}



class BpToKSpipi0(PTo3P):
	'''
	PWA model for :math:`B^+ \\to K_S^0 \\pi^+ \\pi^0`

	The order of the particles is as given by the above reaction.

	The spin analyzer, i.e. the direction of the particle that defines :math:`\\cos(\\theta_H)` is:

	- the :math:`\\pi^+` for the :math:`K_S^0 \\pi^+` isobar; isobarSystemName = KstP
	- the :math:`K_S^0` for the :math:`K_S^0 \\pi^0` isobar; isobarSystemName = Kst0
	- the :math:`\\pi^0` for the :math:`\\pi^+ \\pi^0` isobar; isobarSystemName = Rho

	Args:
		p1 (np.ndarray): 4-momentum of the :math:`K_S^0` final-state particle of shape [<4>,<nEvents>]
		p2 (np.ndarray): 4-momentum of the :math:`\\pi^+` final-state particle of shape [<4>,<nEvents>]
		p3 (np.ndarray): 4-momentum of the :math:`\\pi^0` final-state particle of shape [<4>,<nEvents>]
	'''
	m123 = C.M.B
	m1 = C.M.K0
	m2 = C.M.pi
	m3 = C.M.pi0
	isobarSystemNames = {'KstP': '12', 'Kst0': '13', 'Rho': '23'}
