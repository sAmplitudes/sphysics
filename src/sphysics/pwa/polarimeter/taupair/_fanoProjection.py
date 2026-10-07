# coding: utf-8
'''
Created on Monday 28 09 2026
Author: Yannik Fausch
Description: Fano coefficients of tau pairs with the projection method
'''

import numpy as np

from .... import eventselection, lorentz, math, amplitudes
from ...._constants import Constants
from ... import barrierFactors
from ... import twobody

from ._utils import rotate_to_bodyfixed_nrk
from ._polarimeter import get_polarimeter_from_J

def calculateFanoCoeffsProjection(h_pos : np.ndarray, h_neg : np.ndarray) -> tuple:
	'''
	Calculates the Fano Coefficients and their uncertainties for given Polarimeter vectors using the Projection Method.
	The uncertainties are calculated using the formula for the variance.

	:param h_pos: Polarimeter vector of the tau_pos of dimension (4,nEvents)
	:param h_neg: Polarimeter vector of the tau_neg (4,nEvents)
	:param nEvents: Number of events
	:return: b_pos, b_neg, C, unc
	'''
	nEvents = h_pos.shape[1]

	#Generate prefactors of b_pos, b_neg
	h_pos = h_pos[1:, :]
	h_neg = h_neg[1:, :]
	#Generate prefactors of c_ij
	h_c = (h_pos[:, None, :]*h_neg[None, :, :]).reshape(-1, h_pos.shape[1])
	#Collect all prefactors in one array
	h_i = np.concatenate((h_pos, h_neg, h_c), axis = 0)

	# Calculate the Fano Cefficients using MC integration
	fanoCoef_i = 1/(nEvents)*np.sum(h_i, axis = 1).reshape(15,1)

	#Rearrange FC_i to match the Fano Coefficients including normalizations
	fanoCoef_i[:3] = fanoCoef_i[:3] * 3
	fanoCoef_i[3:6] = fanoCoef_i[3:6] * 3
	fanoCoef_i[6:] = fanoCoef_i[6:] * 9

	#Uncertainties according to variance formula
	norm = (4*np.pi)**2
	norm_i = np.concatenate((np.full((6,1), norm/3),  np.full((9,1), norm/9)))

	varfanoCoef_i = norm**2/(norm_i**2*nEvents)*(1/nEvents*np.sum(h_i**2, axis = 1).reshape(15,1) - ((norm_i/norm)*fanoCoef_i)**2)
	uncFanoCoef = np.sqrt(varfanoCoef_i)

	#Rearrange as single Fano Coefficient
	b_pos = fanoCoef_i[:3]
	b_neg = fanoCoef_i[3:6]
	c_ij = fanoCoef_i[6:].reshape(3,3)

	#return the Fano Coefficients and uncertainties
	return b_pos, b_neg, c_ij, uncFanoCoef



def calculateFanoCoeffsPiPi(sample : eventselection.Variables) -> tuple:
	'''
	Calculates the Fano Coeffieciens for t1p (pi-pi) sample and returns its Fano Coefficients.
	The sample is rotated into the nrk-Frame of the tau_pos and subsequently boosted
	to the tau_pos/tau_neg rest frame to calculate the polarimeter vectors.
	The polarimeter vectors are added to the sample as 'h_pos', 'h_neg'.
	The hadronic currents are equal to the corresponding Pions momenta.

	:param sample: sample pi-pi decay containing 4-momenta of all constituents
	:param latex: Prints a Latex string to display rounded Fano Coefficients according to PDG rounding rules as vectors/matrix.
	:return: Fano coeeficients in the order b_pos, b_neg, c_ij
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

	#Generate 4-Vector of Positron
	pPositron = math.zeros_like(sample.tau_pos)
	pPositron[...] = np.array([Constants.M.upsilon_4S_0/2, 0., 0., (Constants.M.upsilon_4S_0**2/4-Constants.M.e**2)**0.5]).reshape(-1,1)

	#Rotate to the nrk-Frame of tau_pos
	tau_pos, pPositron, nu_pos, pi_pos, tau_neg, nu_neg, pi_neg  = rotate_to_bodyfixed_nrk(tau_pos, pPositron, nu_pos, pi_pos, tau_neg, nu_neg, pi_neg)# pylint: disable=unbalanced-tuple-unpacking

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

	#Add h_pos, h_neg to sample if not already present
	if any(var not in sample.getLoadedVariables() for var in ['h_pos', 'h_neg']):
		sample.addVariable('h_pos')
		sample.addVariable('h_neg')

	#Calculate hadronic current and polarimeter vectors
	j_pos = pi_pos
	sample.h_pos = get_polarimeter_from_J(j_pos, np.zeros(sample.tau_pos.shape[1], dtype=bool), tau_pos, nu_pos)

	j_neg = pi_neg
	sample.h_neg = get_polarimeter_from_J(j_neg, np.ones(sample.tau_neg.shape[1], dtype=bool), tau_neg, nu_neg)

	#Calculate FanoCoefficients from given polarimeter vectors using Projection Method
	b_pos, b_neg, c_ij, uncFC = calculateFanoCoeffsProjection(sample.h_pos, sample.h_neg)

	return b_pos, b_neg, c_ij, uncFC



def calculateFanoCoeffsRhoRho(sample : eventselection.Variables) -> tuple:
	'''
	Calculates the polarimeter vectors for t2p (rho-rho) sample and returns its Fano Coefficients.
	The sample is rotated into the nrk-Frame of the tau_pos and subsequently boosted
	to the tau_pos/tau_neg rest frame to calculate the polarimeter vectors.
	The polarimeter vectors are added to the sample as 'h_pos', 'h_neg'.

	:param sample: sample rho-rho decay containing 4-momenta of all constituents
	:param latex: Prints a Latex string to display rounded Fano Coefficients according to PDG rounding rules as vectors/matrix.
	:return: Fano coeeficients in the order b_pos, b_neg, c_ij
	'''
	#Boost to CMS-frame
	p_cms = sample.tau_pos + sample.tau_neg
	boost = lorentz.getBoostToRestFrame(p_cms)
	tau_pos = lorentz.applyBoost(boost, sample.tau_pos)
	nu_pos = lorentz.applyBoost(boost, sample.nu_pos)
	pi_pos = lorentz.applyBoost(boost, sample.pi_pos)
	pi0_pos = lorentz.applyBoost(boost, sample.pi0_pos)
	tau_neg = lorentz.applyBoost(boost, sample.tau_neg)
	nu_neg = lorentz.applyBoost(boost, sample.nu_neg)
	pi_neg = lorentz.applyBoost(boost, sample.pi_neg)
	pi0_neg = lorentz.applyBoost(boost, sample.pi0_neg)

	#Generate 4-Vector of Positron
	pPositron = math.zeros_like(sample.tau_pos)
	pPositron[...] = np.array([Constants.M.upsilon_4S_0/2, 0., 0., (Constants.M.upsilon_4S_0**2/4-Constants.M.e**2)**0.5]).reshape(-1,1)

	#Rotate to the nrk-Frame of tau_pos
	tau_pos, pPositron, nu_pos, pi_pos, pi0_pos, tau_neg, nu_neg, pi_neg, pi0_neg = rotate_to_bodyfixed_nrk(tau_pos, pPositron, nu_pos, # pylint: disable=unbalanced-tuple-unpacking
                                                                                                            pi_pos, pi0_pos, tau_neg, nu_neg, pi_neg, pi0_neg)

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


	#Create model of t2p decay
	rho_770 = amplitudes.LambdaF(amplitudes.resonance.gounariSakurai, Constants.M.pi, Constants.M.pi, 1, Constants.M['rho(770)0'], Constants.G['rho(770)0'], qR=None )
	barrierXCompensation= amplitudes.LambdaF(barrierFactors.compensationFactor_unnormalizedM, m1=Constants.M.pi, m2=Constants.M.pi0)

	model = twobody.tau.Tau2Twobody("tau-2pi_model")

	wave = 'rho_770_+[1+,1-]=[pi-[1,0]pi+]'
	model.addWave('rho_770_+[1+,1-]=[pi-[1,0]pi+]', "wave_1m", rho_770, barrierXCompensation)

	#Add h_pos, h_neg to sample if not already present
	if any(var not in sample.getLoadedVariables() for var in ['h_pos', 'h_neg']):
		sample.addVariable('h_pos')
		sample.addVariable('h_neg')

	#Calculate hadronic current and polarimeter vectors
	j_pos = model.calcTotalHadronicCurrent(pi_pos, pi0_pos, partialWaveAmplitudes={wave: 1.0}, normalized=False)
	sample.h_pos = get_polarimeter_from_J(j_pos, np.zeros(sample.tau_pos.shape[1], dtype=bool), tau_pos, nu_pos)

	j_neg = model.calcTotalHadronicCurrent(pi_neg, pi0_neg, partialWaveAmplitudes={wave: 1.0}, normalized=False)
	sample.h_neg = get_polarimeter_from_J(j_neg, np.ones(sample.tau_neg.shape[1], dtype=bool), tau_neg, nu_neg)

	#Calculate FanoCoefficients from given polarimeter vectors using Projection Method
	b_pos, b_neg, c_ij, uncFC = calculateFanoCoeffsProjection(sample.h_pos, sample.h_neg)

	return b_pos, b_neg, c_ij, uncFC
