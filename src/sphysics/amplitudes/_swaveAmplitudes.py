# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Parameterizations for two-body S-wave amplitudes, Created on Monday 31 10 2022
'''

from __future__ import absolute_import, print_function, division, annotations

import os
import numpy as np
import tensorflow as tf

from .._constants import Constants
from ..kinematics import twobodyBreakupmomentumZeroBelowZero
from .. import math
from ._utils import TabulatedAmplitude



def glass(mass: float|np.ndarray, a: float, r: float, m0: float, g0: float,
          phiF: float, phiR: float, phiRsin: float, F: float, R: float,
		  mK: float = None, mPi: float = None, threshold: float = np.nan,
		  useAddBW : bool = False) -> complex | np.ndarray:
	"""
	Generalized LASS parameterization for :math:`[K\\pi]_S` amplitude

	Taken from The European Physical Journal C 2014 vol: 74 (11) pp: 3026
	Equations 13.2.13 to 13.2.15 and 13.2.4

	:param mass: :math:`[K\\pi]` mass at which the amplitude is evaluated
	:param a: scattering length
	:param r: effective interaction length
	:param m0: mass of :math:`K^*_0(1430)`
	:param g0: width of :math:`K^*_0(1430)`
	:param phiF: phase offset of non-resonant term
	:param phiR: phase offset in exponent of resonant term
	:param phiRsin: phase offset in sin of resonant term
	:param F: magnitude of non-resonant term
	:param R: magnitude of resonant term
	:param mK: Mass of kaon (charged kaon mass taken from constants if not given)
	:param mPi: Mass of pion (charged pion mass taken from constants if not given)
	:param threshold: Above this mass value, the amplitude is set to zero. Defaults to NaN, which means that no threshold is set
	:param useAddBW: When `True`, the threshold is set to the mass where the kpisPelaezRodas amplitude is nearly zero. Overwrites threshold!
	"""

	if mK is None:
		mK = Constants.M.K
	if mPi is None:
		mPi = Constants.M.pi

	q  = twobodyBreakupmomentumZeroBelowZero(mass**2, mK, mPi)
	q0 = twobodyBreakupmomentumZeroBelowZero(m0**2,   mK, mPi)

	g = g0 * m0 / mass * q / q0

	deltaR = math.arctan(m0 * g / (m0 * m0 - mass * mass))

	deltaF = math.arctan(2.0 * a * q / (2.0 + a * r * q * q))

	# make all inputs tensors
	F = math.castToComplex(F)
	deltaF = math.castToComplex(deltaF)
	phiF = math.castToComplex(phiF)
	R = math.castToComplex(R)
	deltaR = math.castToComplex(deltaR)
	phiRsin = math.castToComplex(phiRsin)
	phiR = math.castToComplex(phiR)
	mass = math.castToComplex(mass)
	q = math.castToComplex(q)

	amp = F * math.sin(deltaF + phiF) *    math.exp(1j*math.castToComplex(deltaF + phiF)) \
	    + R * math.sin(deltaR + phiRsin) * math.exp(1j*math.castToComplex(deltaR + phiR)) * math.exp(2j*math.castToComplex(deltaF + phiF))
	amp *= 0.5*mass / q # divide by Kpi phase-space

	# Set new threshold to use BW component for high mass intensity peak
	if useAddBW:
		threshold = 1.700781

	#Only apply threshold if it is set
	if np.isfinite(threshold):
		# Create a mask for upper threshold
		mask = mass > threshold
		# Apply mask to amplitude
		if isinstance(amp, tf.Tensor):
			amp = tf.where(mask, tf.zeros_like(amp), amp)
		else:
			amp[mask] = 0

	return amp


def lass(mass: float|np.ndarray, a: float, r: float, m0: float, g0: float,
         mK: float = None, mPi: float = None, threshold: float = np.nan,
		 useAddBW : bool = False) -> complex|np.ndarray:
	"""
	LASS parameterization for :math:`[K\\pi]_S` amplitude

	:param mass: :math:`[K\\pi]` mass at which the amplitude is evaluated
	:param a: scattering length
	:param r: effective interaction length
	:param m0: mass of :math:`K^*_0(1430)`
	:param g0: width of :math:`K^*_0(1430)`
	:param mK: Mass of kaon (charged kaon mass taken from constants if not given)
	:param mPi: Mass of pion (charged pion mass taken from constants if not given)
	:param threshold: Above this mass value, the amplitude is set to zero. Defaults to NaN, which means that no threshold is set
	:param useAddBW: When `True`, the threshold is set to the mass where the kpisPelaezRodas amplitude is nearly zero. Overwrites threshold!
	"""

	return glass(mass, a, r, m0, g0,
	             phiF = 0.0, phiR = 0.0, phiRsin = 0.0, F = 1.0, R = 1.0,
				 mK = mK, mPi = mPi, threshold = threshold, useAddBW = useAddBW)


_kpisPelaezRodas = TabulatedAmplitude(os.path.join(os.environ['SPHYSICS'], 'data', 'amplitudes', 'kpiS_Pelaez-Rodas_2010.11222.csv'))
def kpisPelaezRodas(mass: float|np.ndarray|tf.Tensor, threshold: float = 1.8, useAddBW : bool = False) -> complex|np.ndarray|tf.Tensor:
	"""
	Amplitudes for isospin :math:`I=1/2` S-wave :math:`K\\pi` scattering amplitude from Pelaez Rodas

	Parameterization taken from Phys.Rept. 969 (2022) 1 (DOI: 10.1016/j.physrep.2022.03.004).
	We use the parameters from the constraint fit to data (CFD).
	Amplitudes are taken from a look-up table created from a Mathematica notebook privately sent by Arkeitz Rodas.
	The lookup table contains values for :math:`K\\pi` masses from threshold up to 6 GeV/c².
	Below the :math:`K\\pi` threshold, the amplitude is set to zero.
	When adding a Breit-Wigner component for a high-mass peak, set `useAddBW` to `True`. If another threshold is
	desired, set it manually and leave `useAddBW` as `False`.

	:param mass: Mass values at which the amplitudes are evaluated
	:param threshold: Above this mass value, the amplitude is set to zero. Defaults to 1.8, the limit used in the paper
	:param useAddBW: When `True`, the threshold is set to the mass where the amplitude is nearly zero. Overwrites threshold!

	:returns: Complex-valued :math:`K\\pi` S-wave scattering amplitude
	"""

	# Make mass an array if it is a float or integer
	if isinstance(mass, (float, int)):
		mass = np.array([mass])

	# Set new threshold to use BW component for high mass intensity peak
	if useAddBW:
		threshold = 1.700781

	# Create a mask for mass below Kpi threshold and upper threshold
	mask = (mass < Constants.M.K + Constants.M.pi) | (mass > threshold)

	# Get amplitude
	amp = _kpisPelaezRodas(mass)

	# Apply mask to amplitude
	if isinstance(amp, tf.Tensor):
		amp = tf.where(mask, tf.zeros_like(amp), amp)
	else:
		amp[mask] = 0

	return amp
