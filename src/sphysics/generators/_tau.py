# coding: utf-8
'''
Created on Monday 14 03 2022
@author: Stefan Wallner
@description: Generators for tau decays
'''

from __future__ import absolute_import, division, print_function

import numpy as _np

from .. import _constants
from .. import math
from .. import lorentz

from ._phaseSpace import twoBodyDecay, NBodyGenerator

Constants = _constants.Constants

def tauToNBody(nEvents: int, s: float, fsMasses: list, mXRange: tuple=None, tauMass: float = None) -> tuple:
	'''
	Generates phase-space distributed events of :math:`\\tau \\to \\nu_\\tau  (1) (2) ... (n)`
	The returned momenta are in the overall center-of-momentum frame, i.e. the rest frame of the :math:`\\tau^+\\tau^-` system.

	:param nEvents: Number of events to generate (number of returned events, not number of attempts)
	:param s: Center of momentum energy squared of the :math:`\\tau\\tau` system
	:param fsMasses: Final-state masses, including neutrino mass as last entry.
	:param mXRange:  range of invariant mass of n-body system
	:param tauMass: Mass of the tau particle. If none given, use :math:`m_\\tau` from constants
	:return: Momenta :math:`p_{\\tau}`, :math:`p_{\\nu_\\tau}`, :math:`p_{1}`, :math:`p_{2}`, ..., :math:`p_{n}`
	'''

	if tauMass is None:
		tauMass = Constants.M.tau
	nEvents = int(nEvents)

	# generate tau momentum in CMS frame
	pTau = twoBodyDecay(_np.full(nEvents, math.sqrt(s)), tauMass, tauMass)[0]

	# generate final-state momenta in tau reference frame
	kwargs = {}
	if mXRange:
		kwargs['limitMasses'] = {1: mXRange}
	generator = NBodyGenerator(tauMass, fsMasses, **kwargs)

	fsMomenta_tauRF = generator.gen(nEvents)

	# boost from tau reference frame to the CMS
	boostToCMS = lorentz.getBoostFromRestFrame(pTau)
	fsMomenta = tuple( math.einsum('ij...,j...->i...',boostToCMS, p) for p in fsMomenta_tauRF )

	return pTau, fsMomenta[-1], *fsMomenta[0:-1]


def tauToThreePi(nEvents: int, s: float, mXRange: tuple=None, tauMass: float = None, fsMasses: list = None) -> tuple:
	'''
	Generates phase-space distributed events of :math:`\\tau \\to \\nu_\\tau \\pi \\pi \\pi` for three charged pions
	The returned momenta are in the overall center-of-momentum frame, i.e. the rest frame of the :math:`\\tau^+\\tau^-` system.

	:param nEvents: Number of events to generate (number of returned events, not number of attempts)
	:param s: Center of momentum energy squared of the :math:`\\tau\\tau` system
	:param mXRange:  range of :math:`3\\pi` masses in which events are generated
	:param tauMass: Mass of the tau particle. If none given, use :math:`m_\\tau` from constants
	:param fsMasses: Final-state masses. If none given, use :math:`[m_\\pi, m_\\pi, m_\\pi, 0.0]`.
	:return: Momenta :math:`p_{\\tau}`, :math:`p_{\\nu_\\tau}`, :math:`p_{\\pi_1}`, :math:`p_{\\pi_2}`, :math:`p_{\\pi_3}`
	'''
	if fsMasses is None:
		fsMasses = [Constants.M.pi, Constants.M.pi, Constants.M.pi, 0.0]
	return tauToNBody(nEvents, s, fsMasses=fsMasses, mXRange=mXRange, tauMass=tauMass)
