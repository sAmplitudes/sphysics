# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Created on Nov 7, 2019
'''


from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import tensorflow as tf
from ..utils import Logger

from ..kinematics import twobodyBreakupmomentumSquared, twobodyPhasespaceComplexContinuation
from .._constants import Constants
from .. import math
from ..pwa.barrierFactors import barrierFactorSquared_QuiggHipple

log = Logger('Res')



def dynamicWidthForBreitWignerS(s, m1, m2, L, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale):
	'''
	Dynamic width of a relativistic Breit-Wigner amplitude including centrifugal barrier factors.

	The dynamic width is zero below the two-body threshold, i.e., for :math:`s < (m_1 + m_2)^2`.

	If `s` is complex valued, this function returns the dynamic width on the physical sheet,
	i.e., on the sheet without a pole.

	:param s: Mass squared at which the Breit-Wigner should be evaluated
	:param m1: Mass of the first particle of the channel considered in the dynamic width
	:param m2: Mass of the second particle of the channel considered in the dynamic width
	:param L: Angular momentum between the two particles of the channel considered in the dynamic width
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor. If None,
						:math:`q^{2L} / q_0^{2L}` is used
	'''
	breakUpMomentum2 = twobodyBreakupmomentumSquared(s, m1, m2)
	breakUpMomentum02 = twobodyBreakupmomentumSquared(m0**2, m1, m2)

	if breakupMomScale is not None:
		barrierFactorSquaredM = barrierFactorSquared_QuiggHipple(L, breakUpMomentum2, breakupMomScale=breakupMomScale)
		barrierFactorSquared0 = barrierFactorSquared_QuiggHipple(L, breakUpMomentum02, breakupMomScale=breakupMomScale)
	else:
		barrierFactorSquaredM = breakUpMomentum2**( L)
		barrierFactorSquared0 = breakUpMomentum02**(L)

	if math.isComplex(s):
		phaseSpace_2 = twobodyPhasespaceComplexContinuation(s, m1, m2)
	else:
		phaseSpace_2 = math.sqrt(breakUpMomentum2/s, where=s>=(m1+m2)**2, whereNotValue=0.0)

	gamma = g0*(phaseSpace_2*barrierFactorSquaredM)/(math.sqrt(breakUpMomentum02)/m0*barrierFactorSquared0)
	return gamma


def breitWignerS(s, m1, m2, L, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale):
	"""
	Breit-Wigner amplitude with single-channel dynamic width.

	:param s: Mass squared at which the Breit-Wigner should be evaluated
	:param m1: Mass of the first particle of the channel considered in the dynamic width
	:param m2: Mass of the second particle of the channel considered in the dynamic width
	:param L: Angular momentum between the two particles of the channel considered in the dynamic width
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor
						If None, :math:`q^{2L} / q_0^{2L}` is used
	"""

	gamma = dynamicWidthForBreitWignerS(s, m1, m2, L, m0, g0, breakupMomScale=breakupMomScale)

	if isinstance(gamma, tf.Tensor):
		bwAmpl = tf.complex(m0*g0, tf.constant(0, dtype=tf.float64))/tf.complex(m0**2 - s,  -m0*gamma)
	else:
		bwAmpl = np.asarray(m0*g0, np.complex128)/(m0**2 - s  -1j*m0*gamma)
	return bwAmpl


def breitWignerAlternativeDynwidthS(s, m1, m2, L, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale):
	"""
	Breit Wigner amplitude with single-channel dynamic width and an alternative parameterization for the dynamic width.

	The dynamic-width parameterization is

	.. math:: \\Gamma(s) = \\Gamma_0 \\frac{q^{2L+1}}{q_0^{2L+1}}

	This corresponds to the "standard" dynamic width :math:`\\Gamma_\\mathrm{std}(s)` (see above) by

	.. math:: \\Gamma(s) = \\frac{\\sqrt{s}}{m_0} \\Gamma_\\mathrm{std}(s)

	This amplitude is sometimes used for the :math:`\\rho(770)` parameterization (see Pitsu and Ross [10.1016/0550-3213(68)90001-1])

	:param s: Mass squared at which the Breit-Wigner should be evaluated
	:param m1: Mass of the first particle of the channel considered in the dynamic width
	:param m2: Mass of the second particle of the channel considered in the dynamic width
	:param L: Angular momentum between the two particles of the channel considered in the dynamic width
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor.
						If None, :math:`q^{2L} / q_0^{2L}` is used
	"""


	gamma = dynamicWidthForBreitWignerS(s, m1, m2, L, m0, g0, breakupMomScale=breakupMomScale)

	if isinstance(gamma, tf.Tensor):
		bwAmpl = tf.cast(m0*g0,dtype=tf.complex128)/tf.complex(m0**2 - s, -math.sqrt(s)*gamma)
	else:
		bwAmpl = np.asarray(m0*g0, np.complex128)/(m0**2 - s  -1j*math.sqrt(s)*gamma)
	return bwAmpl


def multiChannelBreitWignerS(s, m1, m2, L, x, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale):
	"""
	Breit-Wigner amplitude with multi-channel dynamic width

	:param s: Mass squared at which the Breit-Wigner should be evaluated
	:param m1: List of masses of the first particles of the channels considered in the dynamic width
	:param m2: List of masses of the second particles of the channels considered in the dynamic width
	:param L: List of angular momenta between the two particles of the channels considered in the dynamic width
	:param x: Relative branching fractions between the channels considered in the dynamic width. Should add up to 1
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor.
						If None, :math:`q^{2L} / q_0^{2L}` is used
	"""

	if not (isinstance(x, tf.Tensor) or any(isinstance(xi, tf.Tensor) for xi in x) ) or tf.executing_eagerly():
		if not  len(m1) == len(m2) == len(L) == len(x):
			log.raiseException(ValueError,"Parameter lists 'm1', 'm2', 'L', 'x' have different lengths {}!".format((len(m1),len(m2),len(L),len(x))))
		sum_x = math.sum(x) if not any(isinstance(xi, tf.Tensor) for xi in x) else math.sum(tf.convert_to_tensor(x))
		if not math.allclose(sum_x,1.):
			log.raiseException(ValueError,"Relative branching fractions x[i] do not add up to 1, but to {}!".format(sum_x))
	gamma = 0.
	for xi, m1i, m2i, Li in zip(x, m1, m2, L):
		gamma += xi * dynamicWidthForBreitWignerS(s, m1i, m2i, Li, m0, g0, breakupMomScale=breakupMomScale)

	if isinstance(gamma, tf.Tensor):
		bwAmpl = tf.cast(m0*g0,dtype=tf.complex128)/tf.complex(m0**2 - s, -m0*gamma)
	else:
		bwAmpl = np.asarray(m0*g0, np.complex128)/(m0**2 - s  -1j*m0*gamma)
	return bwAmpl


def twoChannelBreitWignerS(s, m1c1, m2c1, Lc1, m1c2, m2c2, Lc2, x, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale): # pylint: disable=invalid-name
	"""
	Breit-Wigner amplitude with two-channel dynamic width

	:param s: Mass squared at which the Breit-Wigner should be evaluated
	:param m1c1: Mass of the first particle of the first channel considered in the dynamic width
	:param m2c1: Mass of the second particle of the first channel considered in the dynamic width
	:param Lc1: Angular momentum between the two particles of the first channel considered in the dynamic width
	:param m1c2: Mass of the first particle of the second channel considered in the dynamic width
	:param m2c2: Mass of the second particle of the second channel considered in the dynamic width
	:param Lc2: Angular momentum between the two particles of the second channel considered in the dynamic width
	:param x: Relative branching fraction of the second decay channel considered in the dynamic width,
			i.e. :math:`\\Gamma_{\\text{total}} / \\Gamma_{\\text{ch. 2}}`
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor.
						If None, :math:`q^{2L} / q_0^{2L}` is used
	"""

	return multiChannelBreitWignerS(s, m1 = [m1c1,m1c2], m2 = [m2c1,m2c2], L = [Lc1,Lc2], x = [1-x,x], m0 = m0, g0 = g0, breakupMomScale = breakupMomScale)


def constantWidthBreitWignerS(s, m0, g0):
	"""
	Breit-Wigner amplitude with constant width

	:param s: Mass squared at which the Breit-Wigner should be evaluated
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	"""

	bwAmpl = math.castToComplex(m0*g0)/math.complex(m0**2 - s, -m0*g0)
	return bwAmpl


def breitWigner(mass, m1, m2, L, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale):
	"""
	Breit-Wigner amplitude with single-channel dynamic width.

	:param mass: Mass at which the Breit-Wigner should be evaluated
	:param m1: Mass of the first particle of the channel considered in the dynamic width
	:param m2: Mass of the second particle of the channel considered in the dynamic width
	:param L: Angular momentum between the two particles of the channel considered in the dynamic width
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor.
						If None, :math:`q^{2L} / q_0^{2L}` is used
	"""

	return breitWignerS(mass**2, m1, m2, L, m0, g0, breakupMomScale=breakupMomScale)



def nonrelativisticBreitWignerS(s, m0, g0):
	"""
	Nonrelativistic Breit-Wigner amplitude

	:param s: Mass squared at which the Breit-Wigner should be evaluated
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	"""

	bwAmpl = math.castToComplex(g0/2.)/math.complex(m0-math.sqrt(s), -g0/2.)
	return bwAmpl


def nonrelativisticBreitWinger(mass, m0, g0):
	"""
	Nonrelativistic Breit-Wigner amplitude

	:param mass: Mass at which the Breit-Wigner should be evaluated
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	"""

	return constantWidthBreitWignerS(mass**2,m0, g0)


def multiChannelBreitWigner(mass, m1, m2, L, x, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale):
	"""
	Breit-Wigner amplitude with multi-channel dynamic width

	:param mass: Mass at which the Breit-Wigner should be evaluated
	:param m1: List of masses of the first particles of the channels considered in the dynamic width
	:param m2: List of masses of the second particles of the channels considered in the dynamic width
	:param L: List of angular momenta between the two particles of the channels considered in the dynamic width
	:param x: Relative branching fraction between the channels considered in the dynamic width. Should add up to 1
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor.
						If None, :math:`q^{2L} / q_0^{2L}` is used
	"""

	return multiChannelBreitWignerS(mass**2, m1, m2, L, x, m0, g0, breakupMomScale = breakupMomScale)


def twoChannelBreitWigner(mass, m1c1, m2c1, Lc1, m1c2, m2c2, Lc2, x, m0, g0, breakupMomScale = Constants.barrierFactorMomentumscale): # pylint: disable=invalid-name
	"""
	Breit-Wigner amplitude with double-channel dynamic width

	:param mass: Mass at which the Breit-Wigner should be evaluated
	:param m1c1: Mass of the first particle of the first channel considered in the dynamic width
	:param m2c1: Mass of the second particle of the first channel considered in the dynamic width
	:param Lc1: Angular momentum between the two particles of the first channel considered in the dynamic width
	:param m1c2: Mass of the first particle of the second channel considered in the dynamic width
	:param m2c2: Mass of the second particle of the second channel considered in the dynamic width
	:param Lc2: Angular momentum between the two particles of the second channel considered in the dynamic width
	:param x: Relative branching fraction between the two channels considered in the dynamic width,
			i.e. :math:`\\Gamma_{\\text{total}} / \\Gamma_{\\text{ch. 2}}`
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	:param breakMomScale: Momentum scaling factor for the centrifugal barrier factor.
						If None, :math:`q^{2L} / q_0^{2L}` is used
	"""

	return twoChannelBreitWignerS(mass**2, m1c1, m2c1, Lc1, m1c2, m2c2, Lc2, x, m0, g0, breakupMomScale = breakupMomScale)


def constantWidthBreitWigner(mass, m0, g0):
	"""
	Breit-Wigner amplitude with constant width

	:param mass: Mass at which the Breit-Wigner should be evaluated
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	"""

	return constantWidthBreitWignerS(mass**2,m0, g0)


def breitWignerNoBarriersupressionFactor(mass, m1, m2, L, m0, g0):
	"""
	Breit-Wigner amplitude with single-channel dynamic width.

	In contrast to `breitWigner`, the high-m suppressing barrier factors are not included here,
	while the :math:`q^{2L+1}` factor is included.
	Such an amplitude parameterization was used, e.g., in the CLEO :math:`\\tau` \\to :math:`3\\pi` analysis.

	:param mass: Mass at which the Breit-Wigner should be evaluated
	:param m1: Mass of the first particle of the channel considered in the dynamic width
	:param m2: Mass of the second particle of the channel considered in the dynamic width
	:param L: Angular momentum between the two particles of the channel considered in the dynamic width
	:param m0: Nominal mass of the Breit-Wigner
	:param g0: Nominal width of the Breit-Wigner
	"""
	return breitWigner(mass, m1, m2, L, m0, g0, breakupMomScale=None)


__gounariSakkuraiFirstCallWithUnequalMasses = True


def gounariSakuraiFS(s: float|np.ndarray, m1: float|np.ndarray, m2: float|np.ndarray,
                   m0: float|np.ndarray, g0: float|np.ndarray) -> float|np.ndarray:

	"""Real part of the self energy of the Gounari Sakurai amplitude (f function)

	:param s: Invariant mass squared of the decaying system
	:param m1: Mass of the first final-state particle
	:param m2: Mass of the second final-state particle
	:param m0: Nominal mass of the resonance
	:param g0: Nominal width of the resonance
	"""

	global __gounariSakkuraiFirstCallWithUnequalMasses # pylint: disable=global-statement
	if m1 != m2 and __gounariSakkuraiFirstCallWithUnequalMasses:
		log.warning("Gounari Sakurai amplitude valid only for final-state particles equal masses, but you are calling it with unequal masses. Using average mass at the moment.")
		__gounariSakkuraiFirstCallWithUnequalMasses = False


	q2  = twobodyBreakupmomentumSquared(s, m1, m2)
	q02 = twobodyBreakupmomentumSquared(m0**2, m1, m2)

	if math.isComplex(s):
		q = math.sqrt(q2)
	else:
		q = math.sqrt(q2, where=s>=(m1+m2)**2, whereNotValue=0.0)
	q0 = math.sqrt(q02)
	mass = math.sqrt(s)

	# use average mass of final state particles as mean
	mFs = 0.5*(m1+m2)

	h  = 2/np.pi*q /mass*math.log((mass+2.*q )/2./mFs)
	h0 = 2/np.pi*q0/m0  *math.log((m0  +2.*q0)/2./mFs)
	hP0 = h0*( 1./(8*q0**2) - 1./2./m0**2 ) + 1./(2*np.pi*m0**2)

	f = g0*m0**2/q0**3*( q2*(h-h0) + q02*hP0*(m0**2-s) )

	return f



def gounariSakuraiS(s: float|np.ndarray, m1: float|np.ndarray, m2: float|np.ndarray, L: int,
                   m0: float|np.ndarray, g0: float|np.ndarray, qR: float = Constants.barrierFactorMomentumscale) -> float|np.ndarray:
	"""Gounari-Sakurai resonance parameterization

	This parameterization is taken from eq. (17) in Z.Phys.C (1997) 15, 76 (DOI: 10.1007/s002880050523) and following equations.

	However, in contrast to this reference, this parameterization includes centrifugal barrier factors
	in the dynamic width according to *Phys. Rev. D* (2011) **112010**, 83.
	The centrifugal barrier factors can be disabled by setting `qR=None`,
	while the :math:`q^{2L+1}` factor is still included.
	This gives the original parameterization (see next reference).
	The original reference is *Phys. Rev. Lett.* (1968) **244**, 21.

	Also, in contrast to the references above which are normalized such that :math:`\\text{gounariSakurai}(0)=1`, we normalize it to :math:`\\text{gounariSakurai}(m_0)=1`.

	:param s: Invariant mass squared of the decaying system
	:param m1: Mass of the first final-state particle
	:param m2: Mass of the second final-state particle
	:param L: Orbital angular momentum
	:param m0: Nominal mass of the resonance
	:param g0: Nominal width of the resonance
	:param qR: Momentum scaling factor for the centrifugal barrier factor.
			If None, no barrier factor is used, while the :math:`q^{2L}` factor is included.
			Defaults to `Constants.barrierFactorMomentumscale`
	"""


	gamma = dynamicWidthForBreitWignerS(s, m1, m2, L, m0, g0, qR)

	f = gounariSakuraiFS(s, m1, m2, m0, g0)

	# original numerator (see function documentation)
	# d = 3./np.pi*mFs**2/q0**2*math.log( (m0+2*q0)/2./mFs) + m0/2./np.pi/q0 - mFs**2*m0/np.pi/q0**3
	# numerator = m0**2 + d*m0*g0
	numerator = m0*g0

	denumminatorRe = m0**2-s + f
	denumminatorIm = -m0*gamma

	if isinstance(gamma, tf.Tensor):
		ampl = tf.cast(numerator,dtype=tf.complex128)/tf.complex(denumminatorRe, denumminatorIm)
	else:
		ampl = np.asarray(numerator, np.complex128)/(denumminatorRe+1j*denumminatorIm)
	return ampl



def gounariSakuraiAlternativeDynwidthS(s: float|np.ndarray, m1: float|np.ndarray, m2: float|np.ndarray, L: int,
                   m0: float|np.ndarray, g0: float|np.ndarray, qR: float = Constants.barrierFactorMomentumscale) -> float|np.ndarray:
	"""
	Gounari-Sakurai resonance parameterization with an alternative dynamic width

	The dynamic-width parameterization is

		:math:`\\Gamma(s) = \\Gamma_0 \\frac{q^{2L+1}}{q_0^{2L+1}}`

	This corresponds to the "standard" dynamic width :math:`\\Gamma_\\mathrm{std}(s)` (see above) by

		:math:`\\Gamma(s) = \\frac{\\sqrt{s}}{m_0} \\Gamma_\\mathrm{std}(s)`

	This amplitude is sometimes used for the :math:`\\rho(770)` parameterization
	(see Pitsu and Ross [10.1016/0550-3213(68)90001-1]).

	This parameterization is taken from Eq. (17) in *Z. Phys. C* (1997) **15**, 76
	(DOI: `10.1007/s002880050523 <https://doi.org/10.1007/s002880050523>`_) and following equations.
	However, in contrast to this reference, this parameterization includes centrifugal barrier factors
	in the dynamic width according to *Phys. Rev. D* (2011) **112010**, 83.

	The centrifugal barrier factors can be disabled by setting `qR=None`,
	while the :math:`q^{2L+1}` factor is still included.
	This gives the original parameterization (see next reference).
	The original reference is *Phys. Rev. Lett.* (1968) **244**, 21.

	Also, in contrast to the references above which are normalized such that
	:math:`\\text{gounariSakurai}(0) = 1`, we normalize it to
	:math:`\\text{gounariSakurai}(m_0) = 1`.

	:param s: Invariant mass squared of the decaying system
	:param m1: Mass of the first final-state particle
	:param m2: Mass of the second final-state particle
	:param L: Orbital angular momentum
	:param m0: Nominal mass of the resonance
	:param g0: Nominal width of the resonance
	:param qR: Momentum scaling factor for the centrifugal barrier factor.
			If None, no barrier factor is used, while the :math:`q^{2L}` factor is included.
			Defaults to `Constants.barrierFactorMomentumscale`
	"""


	gamma = dynamicWidthForBreitWignerS(s, m1, m2, L, m0, g0, qR)

	f = gounariSakuraiFS(s, m1, m2, m0, g0)

	# original numerator (see function documentation)
	# d = 3./np.pi*mFs**2/q0**2*math.log( (m0+2*q0)/2./mFs) + m0/2./np.pi/q0 - mFs**2*m0/np.pi/q0**3
	# numerator = m0**2 + d*m0*g0
	numerator = m0*g0

	denumminatorRe = m0**2-s + f
	denumminatorIm = -math.sqrt(s)*gamma

	if isinstance(gamma, tf.Tensor):
		ampl = tf.cast(numerator,dtype=tf.complex128)/tf.complex(denumminatorRe, denumminatorIm)
	else:
		ampl = np.asarray(numerator, np.complex128)/(denumminatorRe+1j*denumminatorIm)
	return ampl



def gounariSakurai(mass: float|np.ndarray, m1: float|np.ndarray, m2: float|np.ndarray, L: int,
                   m0: float|np.ndarray, g0: float|np.ndarray, qR: float = Constants.barrierFactorMomentumscale) -> float|np.ndarray:
	"""
	Gounari-Sakurai resonance parameterization

	This parameterization is taken from Eq. (17) in *Z. Phys. C* (1997) **15**, 76
	DOI: `10.1007/s002880050523 <https://doi.org/10.1007/s002880050523>`_ and following equations.
	However, in contrast to this reference, this parameterization includes centrifugal barrier factors
	in the dynamic width according to *Phys. Rev. D* (2011) **112010**, 83.

	The centrifugal barrier factors can be disabled by setting `qR=None`,
	while the :math:`q^{2L+1}` factor is still included.
	This gives the original parameterization (see next reference).
	The original reference is *Phys. Rev. Lett.* (1968) **244**, 21.

	Also, in contrast to the references above which are normalized such that
	:math:`\\text{gounariSakurai}(0) = 1`, we normalize it to
	:math:`\\text{gounariSakurai}(m_0) = 1`.

	:param mass: Invariant mass of the decaying system
	:param m1: Mass of the first final-state particle
	:param m2: Mass of the second final-state particle
	:param L: Orbital angular momentum
	:param m0: Nominal mass of the resonance
	:param g0: Nominal width of the resonance
	:param qR: Momentum scaling factor for the centrifugal barrier factor.
			If None, no barrier factor is used, while the :math:`q^{2L}` factor is included.
			Defaults to `Constants.barrierFactorMomentumscale`
	"""

	return gounariSakuraiS(mass**2, m1, m2, L, m0, g0, qR=qR)
