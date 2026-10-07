# coding: utf-8
'''
Created on Monday 28 09 2026
Author: Yannik Fausch
Description: Fano coefficients of tau pairs with the projection method
'''

import numpy as np

from .... import eventselection
from ._polarimeter import calculatePolarimeterVectorPiPi, calculatePolarimeterVectorRhoRho

def calculateFanoCoeffsProjection(h_pos : np.ndarray, h_neg : np.ndarray) -> tuple:
	'''
	Calculates the Fano Coefficients and their uncertainties for given Polarimeter vectors using the Projection Method.
	The uncertainties are calculated using the formula for the variance.

	:param h_pos: Polarimeter vector of the tau_pos of dimension (4,nEvents)
	:param h_neg: Polarimeter vector of the tau_neg (4,nEvents)
	:return: b_pos, b_neg, C, b_pos_unc, b_neg_unc, c_ij_unc
	'''
	nEvents = h_pos.shape[1]

	#Generate prefactors of b_pos, b_neg
	h_pos = h_pos[1:, :]
	h_neg = h_neg[1:, :]
	#Generate prefactors of c_ij
	h_c = (h_pos[:, None, :]*h_neg[None, :, :]).reshape(-1, nEvents)
	#Collect all prefactors in one array
	h_i = np.concatenate((h_pos, h_neg, h_c), axis = 0)

	# Calculate the Fano Cefficients using MC integration
	fanoCoef_i = 1/(nEvents)*np.sum(h_i, axis = 1).reshape(15,1)

	#Rearrange fanoCoef_i to match the Fano Coefficients including normalizations
	fanoCoef_i[:3] = fanoCoef_i[:3] * 3
	fanoCoef_i[3:6] = fanoCoef_i[3:6] * 3
	fanoCoef_i[6:] = fanoCoef_i[6:] * 9

	#Uncertainties according to variance formula
	norm = (4*np.pi)**2
	norm_i = np.concatenate((np.full((6,1), norm/3),  np.full((9,1), norm/9)))

	varfanoCoef_i = norm**2/(norm_i**2*nEvents)*(1/nEvents*np.sum(h_i**2, axis = 1).reshape(15,1) - ((norm_i/norm)*fanoCoef_i)**2)
	uncFanoCoef = np.sqrt(varfanoCoef_i)

	#Rearrange as single Fano Coefficient and uncertainties
	b_pos_unc = uncFanoCoef[:3]
	b_neg_unc = uncFanoCoef[3:6]
	c_ij_unc = uncFanoCoef[6:].reshape(3,3)

	b_pos = fanoCoef_i[:3]
	b_neg = fanoCoef_i[3:6]
	c_ij = fanoCoef_i[6:].reshape(3,3)

	#return the Fano Coefficients and uncertainties
	return b_pos, b_neg, c_ij, b_pos_unc, b_neg_unc, c_ij_unc


def calculateFanoCoeffsPiPi(sample : eventselection.Variables, p_positron: np.ndarray) -> tuple:
	'''
	Calculates the Fano coefficients for the tau to 1 pi decay using the projection method.

	:param sample: Sample of tau to 1 pi decay
	:param p_positron: Positron 4-momentum with shape ``(4, events)``
	:return: Tuple of Fano Coefficients and corresponding uncertainties in the order (B_pos, B_neg, C,
			 B_pos_unc, B_neg_unc, C_unc)
	'''
	h_pos, h_neg = calculatePolarimeterVectorPiPi(sample, p_positron)
	b_pos, b_neg, c_ij, b_pos_unc, b_neg_unc, c_ij_unc = calculateFanoCoeffsProjection(h_pos, h_neg)

	return b_pos, b_neg, c_ij, b_pos_unc, b_neg_unc, c_ij_unc


def calculateFanoCoeffsRhoRho(sample : eventselection.Variables, p_positron: np.ndarray) -> tuple:
	'''
	Calculates the Fano coefficients for the tau to 2 pi decay using the projection method.

	:param sample: Sample of tau to 2 pi decay
	:param p_positron: Positron 4-momentum with shape ``(4, events)``
	:return: Tuple of Fano Coefficients and corresponding uncertainties in the order (B_pos, B_neg, C,
			 B_pos_unc, B_neg_unc, C_unc)
	'''
	h_pos, h_neg = calculatePolarimeterVectorRhoRho(sample, p_positron)
	b_pos, b_neg, c_ij, b_pos_unc, b_neg_unc, c_ij_unc = calculateFanoCoeffsProjection(h_pos, h_neg)

	return b_pos, b_neg, c_ij, b_pos_unc, b_neg_unc, c_ij_unc
