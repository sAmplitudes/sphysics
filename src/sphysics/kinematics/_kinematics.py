# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Created on Nov 7, 2019
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import tensorflow as tf
import scipy.special

from .. import math


def twobodyBreakupmomentumSquared(s, m1, m2):
	'''
	:param s: Invariant mass squared of the two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	:return: Square of the two-body break-up momentum
	'''
	return (s - (m1+m2)**2)*(s - (m1-m2)**2)/4.0/s

def twobodyBreakupmomentumZeroBelowZeroSquared(s, m1, m2):
	'''
	:param s: Invariant mass squared of the two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	:return: Squared two-body break-up momentum. !!! Zero if M < (m1+m2) !!!
	'''
	q2 = twobodyBreakupmomentumSquared(s, m1, m2)
	return math.zeroBelowZero(q2)

def twobodyBreakupmomentumZeroBelowZero(s, m1, m2):
	'''
	:param s: Invariant mass squared of the two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	:return: Two-body break-up momentum. !!! Zero if :math:`M < (m1+m2)` !!!
	'''
	q2 = twobodyBreakupmomentumSquared(s, m1, m2)
	return math.sqrt(q2, where = s >= (m1+m2)**2, whereNotValue=0.0)

def twobodyBreakupmomentum(s, m1, m2):
	'''
	:param s: Invariant mass squared of the two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	:return: Two-body break-up momentum.
	'''
	q2 = twobodyBreakupmomentumSquared(s, m1, m2)
	return math.sqrt(q2)

def twobodyPhasespace(s, m1, m2):
	'''
	:param s: Invariant mass squared of the two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	:return: Two-body phase space volume. !!! Zero if :math:`M < (m1+m2)` !!!
	'''
	return 2.0*math.sqrt(twobodyBreakupmomentumSquared(s, m1, m2)/s, where=s>(m1+m2)**2, whereNotValue=0.0)

def twobodyPhasespaceComplexContinuation(s, m1, m2):
	'''
	:param s: Invariant mass squared of the two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	:return: Two-body phase space volume. Continued into the complex plane for :math: `M < (m1+m2)`, where it becomes purely imaginary.
	'''
	twobodyBreakupMom = twobodyBreakupmomentumSquared(s, m1, m2)
	twobodyBreakupMom = math.complex(twobodyBreakupMom, math.zeros_like(twobodyBreakupMom))
	if not s.dtype in [complex, np.complex64, np.complex128, tf.complex64, tf.complex128]:
		s = math.complex(s, math.zeros_like(s))
	twobodyBreakupMomDivS = twobodyBreakupMom/s
	return 2.0*1j*math.sqrt(-twobodyBreakupMomDivS)

def threebodyPhasespace(s123: float|np.array, m12: float|np.array, m1: float|np.array, m2: float|np.array, m3: float|np.array) -> float|np.array:
	""" Phase-space of the three-particle system differential in m12 and two-body decay angles

	Value is set to zero outside of the pahse-space boundaries

	Args:
		s123 (float | np.array): Invariant mass squared of the (123) system
		m12 (float | np.array): Invariant mass of the (12) system
		m1 (float | np.array): Mass of particle 1
		m2 (float | np.array): Mass of particle 2
		m3 (float | np.array): Mass of particle 3

	Returns:
		float|np.array: Phase-space
	"""
	q123 = twobodyBreakupmomentumSquared(s123, m12, m3)
	q12 = twobodyBreakupmomentumSquared(m12**2, m1, m2)
	inPS = (m12 > (m1+m2))&(s123 > (m12+m3)**2)
	return math.sqrt(q123*q12/s123, where=inPS, whereNotValue=0.)


def fourbodyPhasespace(s1234: float|np.array, m123: float|np.array, m12: float|np.array,
					   m1: float|np.array, m2: float|np.array, m3: float|np.array, m4: float|np.array) -> float|np.array:
	""" Phase-space of the four-particle system differential in m123, m12, and two-body decay angles

	Value is set to zero outside of the pahse-space boundaries

	Args:
		s1234 (float | np.array): Invariant mass squared of the (1234) system
		m123 (float | np.array): Invariant mass of the (123) system
		m12 (float | np.array): Invariant mass of the (12) system
		m1 (float | np.array): Mass of particle 1
		m2 (float | np.array): Mass of particle 2
		m3 (float | np.array): Mass of particle 3
		m4 (float | np.array): Mass of particle 4

	Returns:
		float|np.array: Phase-space
	"""
	q1234 = twobodyBreakupmomentumSquared(s1234, m123, m4)
	q123 = twobodyBreakupmomentumSquared(m123**2, m12, m3)
	q12 = twobodyBreakupmomentumSquared(m12**2, m1, m2)
	inPs = (m12 > (m1+m2))&(m123 > (m12+m3))&(s1234 > (m123+m4)**2)
	return math.sqrt(q1234*q123*q12/s1234, where=inPs, whereNotValue=0.)


def fourbodyPhasespaceInt12(s1234: float|np.array, m123: float|np.array,
						m1: float|np.array, m2: float|np.array, m3: float|np.array, m4: float|np.array,
						nPoints: int = 100) -> float|np.array:
	""" Phase-space of the four-particle system integrated over m12 and differential in m123 and two-body decay angles

	Value is set to zero outside of the pahse-space boundaries

	Args:
		s1234 (float | np.array): Invariant mass squared of the (1234) system
		m123 (float | np.array): Invariant mass of the (123) system
		m1 (float | np.array): Mass of particle 1
		m2 (float | np.array): Mass of particle 2
		m3 (float | np.array): Mass of particle 3
		m4 (float | np.array): Mass of particle 4
		nPoints (int): Number of integration points. n=100 (default) -> rel. precision ~ :math:`10^{-6}`; n=1000 -> rel. precision ~ :math:`10^{-9}`

	Returns:
		float|np.array: Phase-space
	"""
	if isinstance(m123, float):
		m123 = np.array(m123)
	m123 = m123.reshape(-1,1)

	m_range = (np.array(m1+m2).reshape(-1,1), (m123-m3))
	nodes, weights = scipy.special.roots_legendre(nPoints) # pylint: disable=unbalanced-tuple-unpacking
	weights = weights.reshape(1, -1)
	nodes = nodes.reshape(1, -1)
	nodes   = 0.5*(m_range[1]-m_range[0])*nodes + 0.5*(m_range[1]+m_range[0])
	weights = 0.5*(m_range[1]-m_range[0])*weights

	value = fourbodyPhasespace(s1234, m123, nodes, m1, m2, m3, m4)
	return math.sum(value*weights, axis=-1)


def twobodyBreakupmomentumContinuation(s, m1, m2, integral, m0, integral0):
	'''
	Generalization of two-body breakup momentum for unstable particle m1 or m2 using the pseudo-two-body phase-space approximation

	:param s: Invariant mass squared of the pseudo-two-body system
	:param m1: Mass of one particle
	:param m2: Mass of the other particle
	:param integral: Three-body integral of the decayTopology
	:param m0: Mass where the psuedo-two-body and three body phase space are normalized to each other
	:param integral0: Three-body integral where the psuedo-two-body and three body phase space are normalized to each other
	'''
	breakUpMomentum0 = twobodyBreakupmomentumZeroBelowZero(m0**2, m1, m2)
	qCont = breakUpMomentum0 * math.sqrt(s)/m0 * integral/integral0
	return qCont


def bachelorMomentumInIsobarframeSquared(s, m12, m3):
	'''Squared momentum of the bachelor particle (3) in the rest rest frame of the (1)+(2) system
	for a 3-body system (1)+(2)+(3) with mass M

	:param s: Mass squared of the 3-body system
	:param m12: Mass of the (1)+(2) system
	:param m3: Mass of the bachelor particle (3)
	:return: Squared momentum of (3) in (1)+(2) rest frame
	'''
	energy3 = (s - m12**2 - m3**2)/(2.*m12)
	return energy3**2 - m3**2

def bachelorMomentumInIsobarframe(s, m12, m3):
	'''
	Momentum of the bachelor particle (3) in the rest rest frame of the (1)+(2) system
	for a 3-body system (1)+(2)+(3) with mass M

	:param s: Mass squared of the 3-body system
	:param m12: Mass of the (1)+(2) system
	:param m3: Mass of the bachelor particle (3)
	:return: Momentum of (3) in (1)+(2) rest frame
	'''
	return math.sqrt(bachelorMomentumInIsobarframeSquared(s, m12, m3))

def dalitzS23Limits(s123, s12, m1, m2, m3):
	'''
	Calculate the :math:`s_{23}` kinematic limits for a three-body decay, following the PDG convention (see
	`PDG Review of Particle Physics, Chapter 49 <https://journals.aps.org/prd/pdf/10.1103/PhysRevD.110.030001>`_).

	:param s123: Squared mass of the (123) system
	:param s12: Squared mass of the (1)+(2) system
	:param m1: Mass of particle (1)
	:param m2: Mass of particle (2)
	:param m3: Mass of particle (3)
	:return: Tuple of (s23_min, s23_max), the lower and upper limits of :math:`s_{23}`
	'''
	energy2Numerator = s12 - m1**2 + m2**2
	energy3Numerator = s123 - s12 - m3**2

	s23_max = (((energy2Numerator + energy3Numerator)**2)/(4.0*s12)) - (math.sqrt((energy2Numerator**2)/(4.0*s12) - m2**2) - math.sqrt((energy3Numerator**2)/(4.0*s12) - m3**2))**2
	s23_min = (((energy2Numerator + energy3Numerator)**2)/(4.0*s12)) - (math.sqrt((energy2Numerator**2)/(4.0*s12) - m2**2) + math.sqrt((energy3Numerator**2)/(4.0*s12) - m3**2))**2

	return s23_min, s23_max
