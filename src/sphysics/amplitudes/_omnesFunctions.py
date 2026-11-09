# coding: utf-8
'''
:Author: Leon Heuser <heuser@hiskp.uni-bonn.de>
:Description: Omnès functions
'''
#pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import functools

import numpy as np

from dispersionrelations.integrals import OmnesFunction

from .._constants import Constants as C


def _conformalVariable(s, s0):
	r"""Conformal variable

	.. math:: w(s;s_0)=\frac{\sqrt{s}-\sqrt{s_0-s}}{\sqrt{s}+\sqrt{s_0-s}}

	:param s: Squared mass :math:`s` in GeV²
	:param s0: Parameter :math:`s_0` of the conformal variable in GeV²
	:returns: Real part of :math:`w(s;s_0)`
	"""
	return np.real((np.sqrt(s) - np.sqrt(s0 - s + 0j)) / (np.sqrt(s) + np.sqrt(s0 - s + 0j)))


def piPiSwaveI2_phaseShiftS(s, B0=-80.2, B1=-64.0, z2=0.147, Bh2=340.0, Bh3=-2489.0):
	r"""Phase shift :math:`\delta_0^2(s)` of the isospin :math:`I=2` :math:`\pi\pi` S-wave

	:math:`\delta_0^2` is repulsive (negative) and has no resonance. Up to :math:`\sqrt{s_c}=2.1` GeV we use the parametrisation of
	`Pelaez:2024uav <https://inspirehep.net/literature/2861868>`__ (J.R. Peláez, P. Rabán, J. Ruiz de Elvira, *Phys. Rev. D* **111** (2025) 074003),
	which has a cubic term :math:`\hat B_3`, :math:`\sqrt{s_h}=2.3` GeV, and a fit range up to 2.1 GeV.
	Above :math:`s_c` the phase is continued smoothly to zero with an ad-hoc :math:`1/s` fall-off;
	this continuation is **not** part of the published parametrisations.
	The three pieces, with :math:`k=\sqrt{s/4-M_\pi^2}` and the conformal variable

	.. math:: w(s;s_0)=\frac{\sqrt{s}-\sqrt{s_0-s}}{\sqrt{s}+\sqrt{s_0-s}} .

	.. list-table::
		:header-rows: 1

		* - region
		  - form
		* - :math:`\sqrt{s}<0.85` GeV
		  - :math:`\cot\delta = \dfrac{\sqrt{s}}{2k}\,\dfrac{M_\pi^2}{s-2z_2^2}\,\big[B_0+B_1\,w(s;s_l)\big]`, :math:`\quad\sqrt{s_l}=1.05` GeV
		* - :math:`0.85<\sqrt{s}<2.1` GeV
		  - same prefactor, polynomial :math:`\sum_{i=0}^{3}\hat B_i\,\big[w(s;s_h)-w(s_m;s_h)\big]^i`, :math:`\quad\sqrt{s_m}=0.85,\ \sqrt{s_h}=2.3` GeV
		* - :math:`\sqrt{s}>2.1` GeV
		  - :math:`\delta(s)=\delta_0+(\delta(s_c)-\delta_0)\,\dfrac{\lambda+s_c}{\lambda+s}`, :math:`\quad\sqrt{s_c}=2.1` GeV

	The factor :math:`s-2z_2^2` places the Adler zero (below threshold) at :math:`s=2z_2^2`.

	:math:`\hat B_0,\hat B_1` are fixed by matching the low-energy form at :math:`s_m`;
	:math:`\delta(s_c)` and :math:`\lambda` are fixed by continuity of :math:`\delta` and :math:`\delta'` at :math:`s_c`,
	so that :math:`\delta\to\delta_0` (:math:`=0` for S2) for :math:`s\to\infty`.

	The free parameters are :math:`B_0, B_1, z_2, \hat B_2, \hat B_3`. The default values are the Global II values of
	`Pelaez:2024uav <https://inspirehep.net/literature/2861868>`_ (J.R. Peláez, P. Rabán, J. Ruiz de Elvira, Phys. Rev. D 111 (2025) 074003).

	.. figure:: /figures/amplitudes/piPiSwaveI2_phaseShiftS.png
		:alt: Isospin I=2 pi pi S-wave phase shift as a function of sqrt(s)

		Phase shift :math:`\delta_0^2` for the default parameters. The dotted lines mark the threshold :math:`2M_\pi`
		and the matching points :math:`\sqrt{s_m}=0.85` GeV and :math:`\sqrt{s_c}=2.1` GeV between the low-energy,
		intermediate, and asymptotic regions.

	All quantities are in GeV. :math:`M_\pi` is the charged-pion mass ``Constants.M.pi``.

	The code is from Leon Heuser <heuser@hiskp.uni-bonn.de>.

	:param s: Squared mass :math:`s` of the :math:`\pi\pi` system in GeV². Real-valued `float` or `np.ndarray`.
	:param B0: Low-energy parameter :math:`B_0`
	:param B1: Low-energy parameter :math:`B_1`
	:param z2: Parameter :math:`z_2` in GeV, which places the Adler zero at :math:`s=2z_2^2`
	:param Bh2: Intermediate-energy parameter :math:`\hat B_2`
	:param Bh3: Intermediate-energy parameter :math:`\hat B_3`
	:returns: Phase shift :math:`\delta_0^2(s)` in rad as `np.ndarray`; zero below threshold :math:`s_\text{thr}=4M_\pi^2`.
	"""
	S_THR = 4 * C.M.pi**2                                # pi pi threshold
	SQ_SL, SQ_SM, SQ_SH, SQ_SINF = 1.05, 0.85, 2.3, 2.1     # sqrt(s) of the matching points [GeV]
	S_L, S_M, S_H, S_INF = SQ_SL**2, SQ_SM**2, SQ_SH**2, SQ_SINF**2

	s = np.asarray(s, dtype=float)

	def cot_delta(s):
		k = np.sqrt(np.maximum(s / 4 - C.M.pi**2, 0))
		prefactor = np.sqrt(s) / (2 * k) * C.M.pi**2 / (s - 2 * z2**2)
		# low energy: two-term conformal expansion
		poly_low = B0 + B1 * _conformalVariable(s, S_L)
		# intermediate energy: cubic in the shifted high-energy conformal variable
		dw = _conformalVariable(s, S_H) - _conformalVariable(S_M, S_H)
		Bh0 = B0 + B1 * _conformalVariable(S_M, S_L)
		Bh1 = (B1 * S_L / S_H * np.sqrt(S_H - S_M) / np.sqrt(S_L - S_M)
		       * ((SQ_SM + np.sqrt(S_H - S_M)) / (SQ_SM + np.sqrt(S_L - S_M)))**2)
		poly_mid = Bh0 + Bh1 * dw + Bh2 * dw**2 + Bh3 * dw**3
		return prefactor * np.where(s < S_M, poly_low, poly_mid)

	# high energy: delta(s_inf) (lam + s_inf) / (lam + s), continuous with continuous derivative
	eps = 1e-6
	d_inf = np.arctan(1 / cot_delta(S_INF))
	d_prime = (d_inf - np.arctan(1 / cot_delta(S_INF - eps))) / eps
	lam = -d_inf / d_prime - S_INF

	with np.errstate(divide="ignore", invalid="ignore"):
		delta = np.where(s < S_INF, np.arctan(1 / cot_delta(s)), d_inf * (lam + S_INF) / (lam + s))
	return np.where(s > S_THR, delta, 0.0)



@functools.cache
def _piPiSwaveI2_omnes() -> OmnesFunction:
	"""Build the Omnès function of the isospin I=2 pi pi S-wave once and cache it.

	The phase is evaluated on the integration grid at construction.
	"""
	return OmnesFunction(
		piPiSwaveI2_phaseShiftS,
		threshold=4 * C.M.pi**2,
		infinity=1e6,
		integration_split_points=[2, 5, 10, 50, 100, 1000],
		integration_order=100,
		subtraction_point=0.0,
		subtraction_constants=[0.0],
	)

def piPiSwaveI2_omnesS(s):
	r"""Omnès function :math:`\Omega_0^2(s)` of the isospin :math:`I=2` :math:`\pi\pi` S-wave

	.. math:: \Omega_0^2(s) = \exp\left(\frac{s}{\pi}\int_{4M_\pi^2}^{\infty}\frac{\delta_0^2(x)\,\mathrm{d}x}{x\,(x-s-i\epsilon)}\right)

	The input phase shift :math:`\delta_0^2(s)` is :func:`piPiSwaveI2_phaseShiftS` with its default parameters,
	i.e. the parametrisation of
	`Pelaez:2024uav <https://inspirehep.net/literature/2861868>`__ (J.R. Peláez, P. Rabán, J. Ruiz de Elvira, *Phys. Rev. D* **111** (2025) 074003)
	up to 2.1 GeV, continued smoothly to zero above (see there).

	The integral is evaluated with ``OmnesFunction`` of the `dispersionrelations <https://dispersionrelations.readthedocs.io/>`_ package,
	using one subtraction at :math:`s=0` with :math:`\log\Omega_0^2(0)=0`, i.e. :math:`\Omega_0^2(0)=1`.

	For real :math:`s` above threshold, the boundary value :math:`\Omega_0^2(s+i\epsilon)` is returned. Complex :math:`s` works as well.

	Below threshold :math:`\Omega_0^2` is real; above it, its phase reproduces :math:`\delta_0^2` exactly.
	Since the phase is negative, :math:`\Omega_0^2` dips below 1 near threshold; as :math:`\delta_0^2\to0` at high energy,
	:math:`\Omega_0^2` tends to a constant.

	.. note::
		The Omnès function is built from the phase shift alone, i.e. it assumes elastic :math:`\pi\pi` scattering,
		for which Watson's theorem fixes the phase of the form factor to :math:`\delta_0^2`.
		The Global II fit of `Pelaez:2024uav <https://inspirehep.net/literature/2861868>`__ to the scarce inelasticity data gives
		:math:`\eta_0^2(s)=1-\epsilon\,(1-\hat s/s)^{3/2}` with :math:`\sqrt{\hat s}=0.915` GeV and :math:`\epsilon\approx0.33`,
		i.e. :math:`\eta_0^2\approx0.98` at 1 GeV, :math:`\approx0.91` at 1.2 GeV, and :math:`\approx0.76` at 2.1 GeV.
		This inelasticity is neglected here. Peláez et al. use the :math:`I=2` S-wave as input to their dispersion
		relations only up to 1.62 GeV; above, their parametrisation is a plain fit to the data, which reach 2.1 GeV.

	All quantities are in GeV.

	The example notebook :doc:`/notebooks/omnes_pipiS_I2` shows the phase shift and the Omnès function.

	The code is from Leon Heuser <heuser@hiskp.uni-bonn.de>.

	.. note::
		Evaluating exactly at the subtraction point :math:`s=0` returns ``nan`` because of a division by zero
		inside ``dispersionrelations``, although :math:`\Omega_0^2(0)=1` by construction.

	:param s: Squared mass :math:`s` of the :math:`\pi\pi` system in GeV²; real (spacelike or timelike) or complex. `float`, `list`, or `np.ndarray`.
	:returns: Complex-valued Omnès function as `np.ndarray` of the same shape as `s`.
	"""
	return _piPiSwaveI2_omnes()(s)
