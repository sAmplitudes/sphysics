# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Blatt-Weisskopf Barrier factor, Created on Thursday 03 11 2022
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from ..._constants import Constants
from ... import kinematics
from ... import math
from ...utils import Logger
from ._bfQuiggHippel import barrierFactorPolynomSquared


log = Logger('BW_Zemach')

def compensationFactorSquared_BlattWeisskopf(L: int, s: float|np.ndarray, m1: float|np.ndarray, m2: float|np.ndarray,
                                               mRes: float|np.ndarray, qRadial: float = Constants.barrierFactorMomentumscale):
	"""Squared centrifugal compensation factor for the decay (12) -> (1) (2) with rel. angular momentum L for the Zemach formalism. ABCS

	Parameterization according to Blatt and Weisskopf 1952.
	Taken from The physics of B Factories; Eur.Phys.J.C (2014) 74:3026, equation 13.2.8.

	These compensation factor models ONLY the high energy suppression due to the angular momentum L,
	not the :math:`q^{2L}` part that typically comes with the Zemach formalism.


	They are normalized to be one at the nominal position of the resonance :math:`m_{Res}`

	Args:
		L (int): Relative orbital angular momentum
		s (float | np.ndarray): Invariant mass squred of the two-body system
		m1 (float | np.ndarray): Mass of particle 1
		m2 (float | np.ndarray): Mass of particle 2
		mRes (float | np.ndarray): Nominal mass of the resonance in the two-body system
		qRadial (float): Reference two-body break-up momentum, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	q2 = kinematics.twobodyBreakupmomentumSquared(s, m1, m2)
	if isinstance(q2, np.ndarray):
		if np.any(q2 < 0.):
			log.raiseException(ValueError, "q^2 = {0} < 0", q2[q2<0])

	qRes2 = kinematics.twobodyBreakupmomentumSquared(mRes**2, m1, m2)
	if isinstance(qRes2, np.ndarray):
		if np.any(qRes2 < 0.):
			log.raiseException(ValueError, "qRes^2 = {0} < 0", qRes2[qRes2<0])


	return barrierFactorPolynomSquared(L, qRes2, qRadial) / barrierFactorPolynomSquared(L, q2, qRadial)

def compensationFactor_BlattWeisskopf(L: int, s: float|np.ndarray, m1: float|np.ndarray, m2: float|np.ndarray,
                                               mRes: float|np.ndarray, qRadial: float = Constants.barrierFactorMomentumscale):
	"""Centrifugal compensation factor for the decay (12) -> (1) (2) with rel. angular momentum L for the Zemach formalism.

	Parameterization according to Blatt and Weisskopf 1952.
	Taken from The physics of B Factories; Eur.Phys.J.C (2014) 74:3026, equation 13.2.8.

	These compensation factor models ONLY the high energy suppression due to the angular momentum L,
	not the :math:`q^{2L}` part that typically comes with the Zemach formalism.


	They are normalized to be one at the nominal position of the resonance :math:`m_{Res}`

	Args:
		L (int): Relative orbital angular momentum
		s (float | np.ndarray): Invariant mass squred of the two-body system
		m1 (float | np.ndarray): Mass of particle 1
		m2 (float | np.ndarray): Mass of particle 2
		mRes (float | np.ndarray): Nominal mass of the resonance in the two-body system
		qRadial (float): Reference two-body break-up momentum, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	return math.sqrt(compensationFactorSquared_BlattWeisskopf(L, s, m1, m2, mRes, qRadial=qRadial), where=s>=(m1+m2)**2, whereNotValue=0.0)



def compensationFactorSquared_DalitzBlattWeisskopf(L: int, s: float|np.ndarray, m123: float|np.ndarray, m3: float|np.ndarray,
                                                   mRes: float|np.ndarray, qRadial: float = Constants.barrierFactorMomentumscale):
	"""Squared centrifugal compensation factor for the decay (123) -> (12) (3) with rel. angular momentum L for the Zemach formalism.

	:IMPORTANT:
	Here, the momentum :math:`p_3` of the isobar particle (3) in the (12) rest frame is used instead of the two-body break-up momentum,
	i.e. the momentum :math:`p_3^*` of (3) in the (123) rest frame.
	This compensates the momentum dependence introduced by the Zemach formalism, which is :math:`p_3^{*2L}` not :math:`q^{2L}`.
	This compensation factor should only be used for the (123) -> (12) (3) decay, NOT for the (12) -> (1) (2) decay!

	Parameterization according to Blatt and Weisskopf 1952.
	Taken from The physics of B Factories; Eur.Phys.J.C (2014) 74:3026, equation 13.2.8.

	These compensation factor models ONLY the high energy suppression due to the angular momentum L,
	not the :math:`p_3^{2L}` part that typically comes with the Zemach formalism.


	They are normalized to be one at the nominal position of the resonance :math:`m_{Res}`

	Args:
		L (int): Relative orbital angular momentum
		s (float | np.ndarray): Invariant mass squared of the two-body system (12)
		m123 (float | np.ndarray): Invariant mass of the (123) system
		m3 (float | np.ndarray): Mass of particle 3
		mRes (float | np.ndarray): Nominal mass of the resonance in the two-body system
		qRadial (float): Reference two-body break-up momentum, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	p32 = kinematics.bachelorMomentumInIsobarframeSquared(m123**2, math.sqrt(s), m3)
	if isinstance(p32, np.ndarray):
		if np.any(p32 < 0.):
			log.raiseException(ValueError, "p_3^2 = {0} < 0", p32[p32<0])

	p3Res2 = kinematics.bachelorMomentumInIsobarframeSquared(m123**2, mRes, m3)
	if isinstance(p3Res2, np.ndarray):
		if np.any(p3Res2 < 0.):
			log.raiseException(ValueError, "qRes^2 = {0} < 0", p3Res2[p3Res2<0])

	return barrierFactorPolynomSquared(L, p3Res2, qRadial) / barrierFactorPolynomSquared(L, p32, qRadial)


def compensationFactor_DalitzBlattWeisskopf(L: int, s: float|np.ndarray, m123: float|np.ndarray, m3: float|np.ndarray,
                                               mRes: float|np.ndarray, qRadial: float = Constants.barrierFactorMomentumscale):
	"""Centrifugal compensation factor for the decay (123) -> (12) (3) with rel. angular momentum L for the Zemach formalism.

	:IMPORTANT:
	Here, the momentum :math:`p_3` of the isobar particle (3) in the (12) rest frame is used instead of the two-body break-up momentum,
	i.e. the momentum :math:`p_3^*` of (3) in the (123) rest frame.
	This compensates the momentum dependence introduced by the Zemach formalism, which is :math:`p_3^{*2L}` not :math:`q^{2L}`.
	This compensation factor should only be used for the (123) -> (12) (3) decay, NOT for the (12) -> (1) (2) decay!

	Parameterization according to Blatt and Weisskopf 1952.
	Taken from The physics of B Factories; Eur.Phys.J.C (2014) 74:3026, equation 13.2.8.

	These compensation factor models ONLY the high energy suppression due to the angular momentum L,
	not the :math:`p_3^{2L}` part that typically comes with the Zemach formalism.


	They are normalized to be one at the nominal position of the resonance :math:`m_{Res}`

	Args:
		L (int): Relative orbital angular momentum
		s (float | np.ndarray): Invariant mass squared of the two-body system (12)
		m123 (float | np.ndarray): Invariant mass of the (123) system
		m3 (float | np.ndarray): Mass of particle 3
		mRes (float | np.ndarray): Nominal mass of the resonance in the two-body system
		qRadial (float): Reference two-body break-up momentum, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	return math.sqrt(compensationFactorSquared_DalitzBlattWeisskopf(L, s, m123, m3, mRes, qRadial=qRadial), where=m123>=(math.sqrt(s)+m3), whereNotValue=0.0)


def compensationFactor_unnormalized(L:int, s:float|np.ndarray, m1:float|np.ndarray, m2:float|np.ndarray, breakupMomScale: float = Constants.barrierFactorMomentumscale):
	"""Unnormalized centrifugal compensation factor for the decay (12) -> (1) (2) with rel. angular momentum L

	Args:
		L (int): Relative orbital angular momentum
		s (float | np.ndarray): Invariant mass squared of the two-body system (12)
		m1 (float | np.ndarray): Mass of particle 1
		m2 (float | np.ndarray): Mass of particle 2
		breakupMomScale (float): Reference two-body break-up momentum, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	breakupMom2= kinematics.twobodyBreakupmomentumSquared(s, m1, m2)
	polynomialSquared= barrierFactorPolynomSquared(L,breakupMom2, breakupMomScale=breakupMomScale)
	compensationFactor= 1/math.sqrt(polynomialSquared)
	return compensationFactor


def compensationFactor_unnormalizedM(L:int, M:float|np.ndarray, m1:float|np.ndarray, m2:float|np.ndarray, breakupMomScale: float = Constants.barrierFactorMomentumscale):
	"""Unnormalized centrifugal compensation factor for the decay (12) -> (1) (2) with rel. angular momentum L,
	as a function of M instead of s.

	Args:
		L (int): Relative orbital angular momentum
		M (float | np.ndarray): Invariant mass of the two-body system (12)
		m1 (float | np.ndarray): Mass of particle 1
		m2 (float | np.ndarray): Mass of particle 2
		breakupMomScale (float): Reference two-body break-up momentum, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	return compensationFactor_unnormalized(L, M**2, m1, m2, breakupMomScale=breakupMomScale)
