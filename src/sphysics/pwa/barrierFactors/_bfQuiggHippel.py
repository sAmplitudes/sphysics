# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Created on Thursday 04 05 2023
'''

from __future__ import absolute_import, print_function, division, annotations


from ..._constants import Constants
from ... import math
from ...utils import Logger


log = Logger('BW_Zemach')


def barrierFactorPolynomSquared(L: int, breakupMom2: float, breakupMomScale : float = Constants.barrierFactorMomentumscale) -> float:
	"""Polynom, i.e. denominator of squared centrifugal barrier factors for the decay (12) -> (1) (2) with rel. angular momentum L

	This is the polynomial appearing in the denominator of the squared centrifugal barrier factor

	Parameterization according to  Quigg and Hipple (1972)
	Taken from Ketzer, `Prog.Part.Nucl.Phys. 113 (2020) 103755 <https://doi.org/10.1016/j.ppnp.2020.103755>`_ .


	Args:
		L (int): Relative orbital angular momentum
		breakupMom2 (float| np.ndarray): Two-body break-up momentum !!SQUARED!!
		breakupMomScale (float| np.ndarray): Momentum scale, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""

	z   = (breakupMom2) / (breakupMomScale * breakupMomScale)
	pol = None
	if L == 0:
		pol = math.ones_like(z)
	elif L == 1:
		pol = z + 1
	elif L == 2:
		pol = z * (z + 3) + 9
	elif L == 3:
		pol = z * (z * (z + 6) + 45) + 225
	elif L == 4:
		pol = z * (z * (z * (z + 10) + 135) + 1575) + 11025
	elif L == 5:
		pol = z * (z * (z * (z * (z + 15) + 315) + 6300) + 99225) + 893025
	elif L == 6:
		pol = z * (z * (z * (z * (z * (z + 21) + 630) + 18900) + 496125) + 9823275) + 108056025
# 	elif L == 7: # check long long in python
# 		z3 = z * z * z
# 		pol = (z * (z * (z * (z * (z * (z * (z + 28) + 1134) + 47250) + 1819125) + 58939650) + 1404728325L) + 18261468225LL)
	else:
		log.critical("calculation of Blatt-Weisskopf barrier factor is not (yet) implemented for L = {L}", L=L)

	return pol




def barrierFactorSquared_QuiggHipple(L: int, breakupMom2: float, breakupMomScale : float = Constants.barrierFactorMomentumscale) -> float:
	"""Squared centrifugal barrier factors for the decay (12) -> (1) (2) with rel. angular momentum L

	Parameterization according to  Quigg and Hipple (1972)
	Taken from Ketzer, `Prog.Part.Nucl.Phys. 113 (2020) 103755 <https://doi.org/10.1016/j.ppnp.2020.103755>`_ .

	These barrier factors model angular-momentum barrier effect (:math:`q^{2L}`) AND the high energy suppression due to the angular momentum L.

	They are normalized to be one when the breakup momentum equals the breakupMomScale

	Args:
		L (int): Relative orbital angular momentum
		breakupMom2 (float| np.ndarray): Two-body break-up momentum !!SQUARED!!
		breakupMomScale (float| np.ndarray): Momentum scale, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	z   = breakupMom2 / breakupMomScale**2
	bf2 = None
	if L == 0:
		bf2 = math.ones_like(z)
	elif L == 1:
		bf2 = (2 * z) / barrierFactorPolynomSquared(L,breakupMom2, breakupMomScale)
	elif L == 2:
		bf2 = (13 * z * z) / barrierFactorPolynomSquared(L,breakupMom2, breakupMomScale)
	elif L == 3:
		bf2 = (277 * z * z * z) / barrierFactorPolynomSquared(L,breakupMom2, breakupMomScale)
	elif L == 4:
		z2 = z * z
		bf2 = (12746 * z2 * z2) / barrierFactorPolynomSquared(L,breakupMom2, breakupMomScale)
	elif L == 5:
		z2 = z * z
		bf2 = (998881 * z2 * z2 * z) / barrierFactorPolynomSquared(L,breakupMom2, breakupMomScale)
	elif L == 6:
		z3 = z * z * z
		bf2 = (118394977 * z3 * z3) / barrierFactorPolynomSquared(L,breakupMom2, breakupMomScale)
# 	elif L == 7: # check long long in python
# 		z3 = z * z * z
# 		bf2 = (19727003738LL * z3 * z3 * z) / barrierFactor_QuiggHipple_Polynom_Squared(L,breakupMom, breakupMomScale)
	else:
		log.critical("calculation of Blatt-Weisskopf barrier factor is not (yet) implemented for L = {L}", L=L)
	return bf2


def barrierFactor_QuiggHipple(L: int, breakupMom2: float, breakupMomScale : float = Constants.barrierFactorMomentumscale) -> float:
	"""Centrifugal barrier factors for the decay (12) -> (1) (2) with rel. angular momentum L

	Parameterization according to  Quigg and Hipple (1972)
	Taken from Ketzer, `Prog.Part.Nucl.Phys. 113 (2020) 103755 <https://doi.org/10.1016/j.ppnp.2020.103755>`_ .

	These barrier factors model angular-momentum barrier effect (:math:`q^{2L}`) AND the high energy suppression due to the angular momentum L.

	They are normalized to be one when the breakup momentum equals the breakupMomScale

	Args:
		L (int): Relative orbital angular momentum
		breakupMom2 (float| np.ndarray): Two-body break-up momentum !!SQUARED!!
		breakupMomScale (float| np.ndarray): Momentum scale, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	if isinstance(breakupMom2, float):
		return math.sqrt(barrierFactorSquared_QuiggHipple(L, breakupMom2, breakupMomScale=breakupMomScale)) if breakupMom2 >= 0 else 0.0
	return math.sqrt(barrierFactorSquared_QuiggHipple(L, breakupMom2, breakupMomScale=breakupMomScale), where=breakupMom2>=0, whereNotValue=0.0)


def barrierFactor_QuiggHippleComplexContinuation(L: int, breakupMom2: float, breakupMomScale : float = Constants.barrierFactorMomentumscale) -> float:
	"""Centrifugal barrier factors for the decay (12) -> (1) (2) with rel. angular momentum L

	Parameterization according to  Quigg and Hipple (1972)
	Taken from Ketzer, `Prog.Part.Nucl.Phys. 113 (2020) 103755 <https://doi.org/10.1016/j.ppnp.2020.103755>`_ .

	These barrier factors model angular-momentum barrier effect (:math:`q^{2L}`) AND the high energy suppression due to the angular momentum L.

	They are normalized to be one when the breakup momentum equals the breakupMomScale

	They are analytically continued to complex breakup momenta, i.e. they are defined for negative breakupMom2 as well, yielding complex values.

	Args:
		L (int): Relative orbital angular momentum
		breakupMom2 (float| np.ndarray): Two-body break-up momentum !!SQUARED!!
		breakupMomScale (float| np.ndarray): Momentum scale, representing the inverse of the interaction radius, :math:`r = 1fm \\to qR = 0.1973 GeV/c`
	"""
	breakupMom2 = math.complex(breakupMom2, math.zeros_like(breakupMom2))
	return math.sqrt(barrierFactorSquared_QuiggHipple(L, breakupMom2, breakupMomScale=breakupMomScale))
