# coding: utf-8
'''
Created on Friday 27 03 2026
Author: Stefan Wallner
Description: Tau polarimeter calculation utilities
'''

# pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from .... import math, lorentz, eventselection, amplitudes
from ...._constants import Constants
from ... import barrierFactors
from ... import twobody
from ._utils import rotate_to_bodyfixed_nrk


def get_polarimeter_from_J(J: np.ndarray, isParticle: np.ndarray, pTau: np.ndarray, pNu: np.ndarray) -> np.ndarray:
	'''Calculate the spin analyzer vector from the decay current.

	The polarimeter vector is defined in:
	Comput. Phys. Comm. **64**, 275-295 (1991), `doi: 10.1016/0010-4655(91)90038-M <https://doi.org/10.1016/0010-4655(91)90038-M>`_.
	"TAUOLA - a library of Monte Carlo programs to simulate decays of polarized τ leptons"
	and
	Comput. Phys. Comm. **245**, 109153 (2024), `doi: 10.1016/j.cpc.2024.109153 <https://doi.org/10.1016/j.cpc.2024.109153>`_.
	"The polarimeter vector for τ → 3π ν_τ decays"

	The current and momenta have to be given in the proper reference frame.
	The returned polarimeter vector is also defined in this frame.


	:param J: Decay current with shape ``(4, events)``
	:param isParticle: Boolean array indicating particle (True) vs antiparticle (False) with shape ``(events,)``
	:param pTau: Tau 4-momentum with shape ``(4, events)``
	:param pNu: Tau neutrino 4-momentum with shape ``(4, events)``
	:return: Spin analyzer vector (polarimeter vector) with shape ``(4, events)``
	'''
	va = math.full(pTau.shape[1], 1.0, pTau)
	va = math.where(isParticle, va, -1.0*va)

	mTau = math.sqrt(lorentz.lp(pTau, pTau))
	Jconj = math.conjugate(J)

	S = 2.0*( lorentz.lp(Jconj, pNu)*J + lorentz.lp(J, pNu)*Jconj - lorentz.lp(Jconj, J)*pNu )

	S5 = -2.0* math.imag(  math.einsum('mnrs,n,ne,r,re,s,se->me', lorentz.levitCivitaSymbol, lorentz.et, Jconj, lorentz.et, J, lorentz.et, pNu) )

	Stot = S5 - va*S

	w = lorentz.lp(pTau, S - va*S5)

	h = 1.0/(mTau*w) * (mTau**2 * Stot - lorentz.lp(pTau, Stot)*pTau)

	return math.real(h)


def calculatePolarimeterVectorPiPi(sample: eventselection.Variables, p_positron: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	'''
	Calculate the polarimeter vectors for the tau to 1 pi decay.

	:param sample: Sample of tau to 1 pi decay
	:param p_positron: Positron 4-momentum with shape ``(4, events)``
	:return: Tuple Polarimeter vectors with shape ``(4, events)`` in the order (h_pos, h_neg)
	'''
	#Boost to CMS-frame
	pTotal = sample.tau_pos + sample.tau_neg
	boost = lorentz.getBoostToRestFrame(pTotal)
	tau_pos = lorentz.applyBoost(boost, sample.tau_pos)
	nu_pos = lorentz.applyBoost(boost, sample.nu_pos)
	pi_pos = lorentz.applyBoost(boost, sample.pi_pos)
	tau_neg = lorentz.applyBoost(boost, sample.tau_neg)
	nu_neg = lorentz.applyBoost(boost, sample.nu_neg)
	pi_neg = lorentz.applyBoost(boost, sample.pi_neg)
	p_positron = lorentz.applyBoost(boost, p_positron)

	#Rotate to the nrk-Frame of tau_pos
	tau_pos, p_positron, nu_pos, pi_pos, tau_neg, nu_neg, pi_neg  = rotate_to_bodyfixed_nrk(tau_pos, p_positron, # pylint: disable=unbalanced-tuple-unpacking
																							nu_pos, pi_pos, tau_neg, nu_neg, pi_neg)
	#Boost to tau_pos
	boost = lorentz.getBoostToRestFrame(tau_pos)
	tau_pos = lorentz.applyBoost(boost, tau_pos)
	nu_pos = lorentz.applyBoost(boost, nu_pos)
	pi_pos = lorentz.applyBoost(boost, pi_pos)

	#Boost to tau_neg
	boost = lorentz.getBoostToRestFrame(tau_neg)
	tau_neg = lorentz.applyBoost(boost, tau_neg)
	nu_neg = lorentz.applyBoost(boost, nu_neg)
	pi_neg = lorentz.applyBoost(boost, pi_neg)

	#Calculate hadronic current and polarimeter vectors
	j_pos = -1j*pi_pos
	h_pos = get_polarimeter_from_J(j_pos, np.zeros(sample.tau_pos.shape[1], dtype=bool), tau_pos, nu_pos)

	j_neg = 1j*pi_neg
	h_neg = get_polarimeter_from_J(j_neg, np.ones(sample.tau_neg.shape[1], dtype=bool), tau_neg, nu_neg)

	return h_pos, h_neg


def calculatePolarimeterVectorRhoRho(sample: eventselection.Variables, p_positron: np.ndarray) -> tuple:
	'''
	Calculate the polarimeter vectors for the tau to 2 pi decay.

	:param sample: Sample of tau to 2 pi decay
	:param p_positron: Positron 4-momentum with shape ``(4, events)``
	:return: Tuple Polarimeter vectors with shape ``(4, events)`` in the order (h_pos, h_neg)
	'''
	#Boost to CMS-frame
	pTotal = sample.tau_pos + sample.tau_neg
	boost = lorentz.getBoostToRestFrame(pTotal)
	tau_pos = lorentz.applyBoost(boost, sample.tau_pos)
	nu_pos = lorentz.applyBoost(boost, sample.nu_pos)
	pi_pos = lorentz.applyBoost(boost, sample.pi_pos)
	pi0_pos = lorentz.applyBoost(boost, sample.pi0_pos)
	tau_neg = lorentz.applyBoost(boost, sample.tau_neg)
	nu_neg = lorentz.applyBoost(boost, sample.nu_neg)
	pi_neg = lorentz.applyBoost(boost, sample.pi_neg)
	pi0_neg = lorentz.applyBoost(boost, sample.pi0_neg)
	p_positron = lorentz.applyBoost(boost, p_positron)

	#Rotate to the nrk-Frame of tau_pos
	tau_pos, p_positron, nu_pos, pi_pos, pi0_pos, tau_neg, nu_neg, pi_neg, pi0_neg = rotate_to_bodyfixed_nrk(tau_pos, p_positron, # pylint: disable=unbalanced-tuple-unpacking
																											 nu_pos, pi_pos, pi0_pos, tau_neg, nu_neg, pi_neg, pi0_neg)

	#Boost to tau_pos
	boost = lorentz.getBoostToRestFrame(tau_pos)
	tau_pos = lorentz.applyBoost(boost, tau_pos)
	nu_pos = lorentz.applyBoost(boost, nu_pos)
	pi_pos = lorentz.applyBoost(boost, pi_pos)
	pi0_pos = lorentz.applyBoost(boost, pi0_pos)

	#Boost to tau_neg
	boost = lorentz.getBoostToRestFrame(tau_neg)
	tau_neg = lorentz.applyBoost(boost, tau_neg)
	nu_neg = lorentz.applyBoost(boost, nu_neg)
	pi_neg = lorentz.applyBoost(boost, pi_neg)
	pi0_neg = lorentz.applyBoost(boost, pi0_neg)

	#Model of t2p decay
	rho_770 = amplitudes.LambdaF(amplitudes.resonance.gounariSakurai, Constants.M.pi, Constants.M.pi,
								 1, Constants.M['rho(770)0'], Constants.G['rho(770)0'], qR=None )
	barrierXCompensation= amplitudes.LambdaF(barrierFactors.compensationFactor_unnormalizedM, m1=Constants.M.pi, m2=Constants.M.pi0)
	model = twobody.tau.Tau2Twobody("tau-2pi_model")
	wave = 'rho_770_+[1+,1-]=[pi-[1,0]pi+]'
	model.addWave('rho_770_+[1+,1-]=[pi-[1,0]pi+]', "wave_1m", rho_770, barrierXCompensation)

	#Calculate hadronic current and polarimeter vectors
	j_pos = model.calcTotalHadronicCurrent(pi_pos, pi0_pos, partialWaveAmplitudes={wave: 1.0}, normalized=False)
	h_pos = get_polarimeter_from_J(j_pos, np.zeros(sample.tau_pos.shape[1], dtype=bool), tau_pos, nu_pos)

	j_neg = model.calcTotalHadronicCurrent(pi_neg, pi0_neg, partialWaveAmplitudes={wave: 1.0}, normalized=False)
	h_neg = get_polarimeter_from_J(j_neg, np.ones(sample.tau_neg.shape[1], dtype=bool), tau_neg, nu_neg)

	return h_pos, h_neg
