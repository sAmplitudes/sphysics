#!/usr/bin/env python
# coding: utf-8
'''
Tests for the phase shift and the Omnès function of the isospin I=2 pi pi S-wave
'''

from __future__ import absolute_import, print_function, division

import unittest

import numpy as np
from scipy.integrate import quad

from dispersionrelations.integrals import OmnesFunction

from sphysics._constants import Constants as C
from sphysics.amplitudes.resonance import piPiSwaveI2_phaseShiftS, piPiSwaveI2_omnesS


S_THR = 4 * C.M.pi**2
S_M, S_C = 0.85**2, 2.1**2  # matching points of the parametrisation [GeV^2]

# Parameters pinned explicitly, so that the regression values below do not depend on the default parameters
PARAMS = dict(B0=-79.3, B1=-65.8, z2=0.1471, Bh2=132.0, Bh3=-2081.0)

# Same numerical settings as in the module
OMNES_SETTINGS = dict(threshold=S_THR, infinity=1e6, integration_split_points=[2, 5, 10, 50, 100, 1000],
                      integration_order=100, subtraction_point=0.0, subtraction_constants=[0.0])


class PiPiSwaveI2PhaseShiftTest(unittest.TestCase):

	def test_zero_and_finite_below_threshold(self):
		# includes negative s, s = 0, the Adler zero s = 2 z_2^2, and the threshold itself
		s = np.array([-1.0, 0.0, 2 * PARAMS['z2']**2, 0.5 * S_THR, S_THR])
		np.testing.assert_array_equal(piPiSwaveI2_phaseShiftS(s, **PARAMS), 0.0)

	def test_negative_bounded_and_vanishing_at_high_energy(self):
		s = np.linspace(1.0001 * S_THR, 100.0, 20000)
		delta = piPiSwaveI2_phaseShiftS(s, **PARAMS)
		self.assertTrue(np.all(np.isfinite(delta)))
		self.assertTrue(np.all(delta < 0))
		self.assertTrue(np.all(delta > -np.pi / 2))
		self.assertLess(abs(piPiSwaveI2_phaseShiftS(1e4, **PARAMS)), 1e-4)
		# 1/s fall-off above s_c: delta * s approaches a constant
		ratio = (piPiSwaveI2_phaseShiftS(1e3, **PARAMS) * 1e3) / (piPiSwaveI2_phaseShiftS(1e4, **PARAMS) * 1e4)
		self.assertAlmostEqual(ratio, 1.0, delta=1e-2)

	def test_continuous_with_continuous_derivative_at_matching_points(self):
		h = 1e-5
		for s0 in (S_M, S_C):
			below, at, above = piPiSwaveI2_phaseShiftS([s0 - 1e-9, s0, s0 + 1e-9], **PARAMS)
			self.assertAlmostEqual(below, at, 7)
			self.assertAlmostEqual(above, at, 7)
			derivativeBelow = (piPiSwaveI2_phaseShiftS(s0, **PARAMS) - piPiSwaveI2_phaseShiftS(s0 - h, **PARAMS)) / h
			derivativeAbove = (piPiSwaveI2_phaseShiftS(s0 + h, **PARAMS) - piPiSwaveI2_phaseShiftS(s0, **PARAMS)) / h
			self.assertAlmostEqual(derivativeAbove / derivativeBelow, 1.0, delta=1e-3)

	def test_threshold_behaviour(self):
		# delta = a k + O(k^3) near threshold, with the scattering length a
		k = np.array([1e-4, 2e-4, 4e-4])
		s = 4 * (C.M.pi**2 + k**2)
		a = piPiSwaveI2_phaseShiftS(s, **PARAMS) / k
		np.testing.assert_allclose(a, a[0], rtol=1e-4)
		# a_0^2 M_pi, cf. -0.042(4) in GarciaMartin:2011cn
		self.assertAlmostEqual(a[0] * C.M.pi, -0.0424, delta=0.0005)

	def test_regression_values(self):
		# values of this implementation; they agree with the notebook implementation of Leon Heuser
		# (which uses M_pi = 0.13957 GeV) to 3e-6
		sqrtS = np.array([0.5, 0.85, 1.0, 1.42, 2.0, 2.1, 2.5, 5.0])
		expected = np.array([-0.1463482609, -0.3515899489, -0.4177805713, -0.4599175256,
		                     -0.2205483429, -0.1715182119, -0.0756878530, -0.0113077311])
		np.testing.assert_allclose(piPiSwaveI2_phaseShiftS(sqrtS**2, **PARAMS), expected, atol=1e-9)

	def test_shapes(self):
		self.assertEqual(np.shape(piPiSwaveI2_phaseShiftS(1.0)), ())
		self.assertEqual(piPiSwaveI2_phaseShiftS([0.5, 1.0, 2.0]).shape, (3,))
		self.assertEqual(piPiSwaveI2_phaseShiftS(np.full((2, 3), 1.0)).shape, (2, 3))


class PiPiSwaveI2OmnesTest(unittest.TestCase):

	def test_normalisation_and_reality_below_threshold(self):
		s = np.concatenate([np.linspace(-5.0, -1e-6, 50), np.linspace(1e-6, S_THR * (1 - 1e-9), 50)])
		omega = piPiSwaveI2_omnesS(s)
		self.assertTrue(np.all(np.isfinite(omega)))
		np.testing.assert_array_equal(omega.imag, 0.0)
		self.assertTrue(np.all(omega.real > 0))
		# Omega(0) = 1 by the subtraction at s = 0
		self.assertAlmostEqual(piPiSwaveI2_omnesS(1e-9).real, 1.0, 6)
		self.assertAlmostEqual(piPiSwaveI2_omnesS(-1e-9).real, 1.0, 6)

	def test_phase_equals_phase_shift_above_threshold(self):
		s = np.linspace(S_THR * (1 + 1e-6), 25.0, 500)
		omega = piPiSwaveI2_omnesS(s)
		self.assertTrue(np.all(np.isfinite(omega)))
		np.testing.assert_allclose(np.angle(omega), piPiSwaveI2_phaseShiftS(s), atol=1e-10)

	def test_tends_to_constant_at_high_energy(self):
		# delta -> 0 at high energy, hence |Omega| -> const
		self.assertAlmostEqual(abs(piPiSwaveI2_omnesS(1e5)) / abs(piPiSwaveI2_omnesS(1e4)), 1.0, delta=1e-3)

	def test_schwarz_reflection_and_shapes(self):
		s = np.array([0.5 + 0.1j, 1.0 - 0.3j, -0.2 + 0.05j])
		np.testing.assert_allclose(piPiSwaveI2_omnesS(np.conj(s)), np.conj(piPiSwaveI2_omnesS(s)), rtol=1e-12)
		self.assertEqual(np.shape(piPiSwaveI2_omnesS(1.0)), ())
		self.assertEqual(piPiSwaveI2_omnesS(np.full((2, 3), 1.0)).shape, (2, 3))

	def test_dispersion_integral_against_independent_quadrature(self):
		# log Omega(s) = s/pi PV int_{s_thr}^inf delta(x) / (x (x - s)) dx + i delta(s), evaluated with scipy
		def logOmega(s):
			integrand = lambda x: piPiSwaveI2_phaseShiftS(x) / x
			if s < S_THR:
				return s / np.pi * quad(lambda x: integrand(x) / (x - s), S_THR, np.inf, limit=500)[0]
			cut = 50.0
			principalValue = quad(integrand, S_THR, cut, weight='cauchy', wvar=s, limit=500)[0] \
			               + quad(lambda x: integrand(x) / (x - s), cut, np.inf, limit=500)[0]
			return s / np.pi * principalValue + 1j * piPiSwaveI2_phaseShiftS(s)
		for s in (-0.5, 0.05, 0.5, 1.0, 2.0, 4.0):
			np.testing.assert_allclose(piPiSwaveI2_omnesS(s), np.exp(logOmega(s)), rtol=2e-5)

	def test_matches_explicit_construction(self):
		# the module wraps exactly this object (also shown in the docs notebook)
		omnes = OmnesFunction(piPiSwaveI2_phaseShiftS, **OMNES_SETTINGS)
		s = np.array([-0.5, 1.0, 0.5 + 0.1j])
		np.testing.assert_allclose(piPiSwaveI2_omnesS(s), omnes(s), rtol=1e-12)

	def test_notebook_reference_values(self):
		# values printed in the notebook 2026-10_omnes_sharing.ipynb of Leon Heuser (M_pi = 0.13957 GeV there)
		omnes = OmnesFunction(lambda s: piPiSwaveI2_phaseShiftS(s, **PARAMS), **OMNES_SETTINGS)
		expected = np.array([1.12578456 + 0j, 0.97145757 - 0.43124529j, 0.93902529 - 0.24640439j])
		np.testing.assert_allclose(omnes([-0.5, 1.0, 0.5 + 0.1j]), expected, atol=2e-5)


if __name__ == '__main__':
	unittest.main()
