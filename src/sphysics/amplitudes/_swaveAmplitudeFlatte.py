# coding: utf-8
'''
Created on Tuesday 03 09 2024
Author: Godo Kurten
Description: Amplitude for [ππ]S wave from Flatte paper. Taken from [M. Ablikim et al, Phys. Let. B607, 243] BES II.
'''
#pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

from .._constants import Constants as C
from ..kinematics import twobodyPhasespaceComplexContinuation
from ._utils import prepare_mass_tensor

imag = 0+1j

def flatteS(s, M0=C.Amplitudes.FlatteF0.m0, g1=C.Amplitudes.FlatteF0.g1, g2g1=C.Amplitudes.FlatteF0.g2g1):
	"""Amplitude for [ππ]S wave from Flatte parameterization.
	[M. Ablikim et al, Phys. Let. B607, 243] BES II

	:param s: Squared mass of the two-body system. Can be `float`, `np.ndarray`, or `tf.Tensor`.
	:param M0: Nominal mass of the resonance (Defaults to :math:`f_0(980)` mass value from paper mentioned above).
	:param g1: Coupling constant to the first channel (Defaults to :math:`\\pi\\pi`-channel value from paper mentioned above).
	:param g2g1: Product of coupling constants of the first and second channel (Defaults to :math:`K\\bar{K} \\times \\pi\\pi`-channel value from paper mentioned above).
	:returns: Complex-valued amplitude as `float`, `np.ndarray`, or `tf.Tensor`.
	"""
	# cast complex
	s = prepare_mass_tensor(s)

	# Phase Space factors
	phaseSpacePi = twobodyPhasespaceComplexContinuation(s, C.M.pi, C.M.pi)
	phaseSpaceK = twobodyPhasespaceComplexContinuation(s, C.M.K, C.M.K)
	denom = M0*M0 - s + 0j
	denom -= imag * g1 * (phaseSpacePi + g2g1*phaseSpaceK)

	# Amplitude
	amp = 1./denom

	return amp

def flatteNormS(s, M0=C.Amplitudes.FlatteF0.m0, g1=C.Amplitudes.FlatteF0.g1, g2g1=C.Amplitudes.FlatteF0.g2g1):
	"""
	Normalized amplitude for ππ]S wave from Flatte parameterization.
	It is normalized such that it is :math:`0 + 1i` at :math:`s = M_0^2`.

	[M. Ablikim et al, Phys. Lett. B607, 243] BES II

	:param s: Squared mass of the two-body system. Can be `float`, `np.ndarray`, or `tf.Tensor`.
	:param M0: Nominal mass of the resonance (Defaults to :math:`f_0(980)` mass value from paper mentioned above).
	:param g1: Coupling constant to the first channel (Defaults to :math:`\\pi\\pi`-channel value from paper mentioned above).
	:param g2g1: Product of coupling constants of the first and second channel (Defaults to :math:`K\\bar{K} \\times \\pi\\pi`-channel value from paper mentioned above).
	:returns: Complex-valued amplitude as `float`, `np.ndarray`, or `tf.Tensor`.
	"""

	# Get amplitude
	amp = flatteS(s, M0, g1, g2g1)

	# Get normalization
	norm = imag / flatteS(M0*M0, M0, g1, g2g1)

	return amp*norm

def flatteNorm(mass, M0=C.Amplitudes.FlatteF0.m0, g1=C.Amplitudes.FlatteF0.g1, g2g1=C.Amplitudes.FlatteF0.g2g1):
	"""Normalized amplitude for [ππ]S wave from Flatte parameterization.
	It is normalized such that it is :math:`0 + 1i` at :math:`\\text{mass} = M_0`.

	[M. Ablikim et al, Phys. Let. B607, 243] BES II

	:param mass: Mass of the two-body system. Can be `float`, `np.ndarray`, or `tf.Tensor`.
	:param M0: Nominal mass of the resonance (Defaults to :math:`f_0(980)` mass value from paper mentioned above).
	:param g1: Coupling constant to the first channel (Defaults to :math:`\\pi\\pi`-channel value from paper mentioned above).
	:param g2g1: Product of coupling constants of the first and second channel (Defaults to :math:`K\\bar{K} \\times \\pi\\pi`-channel value from paper mentioned above).
	:returns: Complex-valued amplitude as `float`, `np.ndarray`, or `tf.Tensor`.
	"""

	return flatteNormS(mass**2, M0, g1, g2g1)

def flatte(mass, M0=C.Amplitudes.FlatteF0.m0, g1=C.Amplitudes.FlatteF0.g1, g2g1=C.Amplitudes.FlatteF0.g2g1):
	"""Amplitude for [ππ]S wave from Flatte parameterization.
	[M. Ablikim et al, Phys. Let. B607, 243] BES II

	:param mass: Mass of the two-body system. Can be `float`, `np.ndarray`, or `tf.Tensor`.
	:param M0: Nominal mass of the resonance (Defaults to :math:`f_0(980)` mass value from paper mentioned above).
	:param g1: Coupling constant to the first channel (Defaults to :math:`\\pi\\pi`-channel value from paper mentioned above).
	:param g2g1: Product of coupling constants of the first and second channel (Defaults to :math:`K\\bar{K} \\times \\pi\\pi`-channel value from paper mentioned above).
	:returns: Complex-valued amplitude as `float`, `np.ndarray`, or `tf.Tensor`.
	"""

	return flatteS(mass**2, M0, g1, g2g1)
