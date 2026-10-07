# coding: utf-8
'''
Created on Friday 23 06 2023
Author: Stefan Wallner
Description: Amplitude for [Kπ]S wave from Palano Pennington paper. Taken from arXiv:1701.04881v1 (an updated version (v2) is no longer available)
'''
#pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import tensorflow as tf
from .. import math

from ..kinematics import twobodyPhasespace



#############################
# parametersm_pi
#############################

# PDG 2017
m_pi = 0.13957061
m_K  = 0.493677
m_eta = 0.547862

PP_C = np.empty((3,3,4), dtype=float) # i=0 and j=0 will not be used (to have the same index as in the paper)
PP_C[:,:,:] = None
PP_Kpoles = ['a', 'b']
PP_sKpi = m_K**2 + m_pi**2
PP_sA = 0.87753*PP_sKpi
PP_s_top = 5.832
PP_s_bot = 0.36
PP_N = 3
PP_g = {'a': [None, 0.3139, -0.00775], 'b': [None, 1.1804, -0.22335]}
PP_sPol = {'a': 1.7991, 'b': 8.3627}
PP_C[1,1,0] = -0.1553
PP_C[1,1,1] = 0.0909
PP_C[1,1,2] = 0.8618
PP_C[1,1,3] = 0.0629
PP_C[1,2,0] = 0.0738
PP_C[1,2,1] = 0.3866
PP_C[1,2,2] = 1.2195
PP_C[1,2,3] = 0.8390
PP_C[2,2,0] = -0.0036
PP_C[2,2,1] = 0.2590
PP_C[2,2,2] = 1.6950
PP_C[2,2,3] = 2.2300
PP_gamma = 2.274


def rho_m1m2(s,m1,m2):
	'''
	Two-body pahase-space
	'''
	return math.castToComplex(twobodyPhasespace(s, m1, m2))

def rho_Kpi(s):
	return math.castToComplex(rho_m1m2(s,m_K,m_pi))

def rho_Keta(s):
	return math.castToComplex(rho_m1m2(s,m_K,m_eta))

def rho(i,s):
	return math.castToComplex(rho_Kpi(s)) if i == 1 else math.castToComplex(rho_Keta(s))


def X(s):
	return math.castToComplex(( 2.0*s-(PP_s_top+PP_s_bot) ) / (PP_s_top - PP_s_bot))

def K(i,j,s):
	x_s = X(s)
	if isinstance(s, tf.Tensor):
		pols = tf.reduce_sum([ PP_g[alpha][i]*PP_g[alpha][j]/(PP_sPol[alpha]-s) for alpha in PP_Kpoles], axis=0)
		poly = tf.reduce_sum([ PP_C[i][j][n]*(x_s**n) for n in range(PP_N+1) ], axis=0)
	else:
		pols = np.sum([ PP_g[alpha][i]*PP_g[alpha][j]/(PP_sPol[alpha]-s) for alpha in PP_Kpoles], axis=0)
		poly = np.sum([ PP_C[i][j][n]*(x_s**n) for n in range(PP_N+1) ], axis=0)
	return math.castToComplex(s - PP_sA)/math.castToComplex(PP_sKpi)*(math.castToComplex(pols) + math.castToComplex(poly))

def detK(s):
	return math.castToComplex(K(1,1,s)*K(2,2,s) - K(1,2,s)**2)

def delta(s):
	return 1.0 - 1j*math.castToComplex(rho(1,s)*K(1,1,s)) - 1j*math.castToComplex(rho(2,s)*K(2,2,s)) - math.castToComplex(rho(1,s)*rho(2,s)*detK(s))



def PP_T_11(s):
	return ( math.castToComplex(K(1,1,s)) - 1j*math.castToComplex(rho(2,s)*detK(s))) / math.castToComplex(delta(s))
def PP_T_12(s):
	return math.castToComplex(K(1,2,s)) / delta(s)
def PP_T_22(s):
	return ( math.castToComplex(K(2,2,s)) - 1j*math.castToComplex(rho(1,s)*detK(s))) / math.castToComplex(delta(s))


def palanoPenningtonKpiKpi(m: np.ndarray|tf.Tensor, threshold: float = None) -> np.ndarray|tf.Tensor:
	"""
	K-matrix parameterization for :math:`[K\\pi]_S` wave from Palano, Pennington

	Taken from arXiv:1701.04881v1 (an updated version (v2) is no longer available).
	Equations 2 to 6.

	Fixed two bugs in the formulas of the original paper:

	- Equation 3: :math:`- \\rho_a \\det(K) \\rightarrow - i \\rho_a \\det(K)`
	- Equation 4: :math:`(s - s_{\\alpha}) \\rightarrow (s_{\\alpha} - s)`

	:param m: Mass of the two-body system
	:param threshold: After the threshold, the amplitude is set to zero

	:returns: Complex-valued :math:`K\\pi` S-wave scattering amplitude
	"""
	s = m**2
	matrix_element = PP_T_11(s)
	if threshold is None:
		return matrix_element
	condition = m>threshold
	matrix_element = math.where(condition, math.zeros_like(PP_T_11(s)), PP_T_11(s))
	return matrix_element



def palanoPenningtonKetaKpi(m : np.ndarray|tf.Tensor, threshold: float = None) -> np.ndarray|tf.Tensor:
	"""
	K-matrix parameterization for :math:`[K\\pi]_S` wave from Palano, Pennington

	Taken from arXiv:1701.04881v1 (an updated version (v2) is no longer available).
	Equations 2 to 6.

	Fixed two bugs in the formulas of the original paper:

	- Equation 3: :math:`- \\rho_a \det(K) \\to - i \\rho_a \\det(K)`
	- Equation 4: :math:`(s - s_{\\alpha}) \\to (s_{\\alpha} - s)`

	:param m: Mass of the two-body system
	:param threshold: After the threshold, the amplitude is set to zero

	:returns: :math:`K\\eta \\to K\\pi` complex-valued amplitude
	"""
	s = m**2
	matrix_element = PP_T_12(s)
	condition = m>threshold
	matrix_element = math.where(condition, math.zeros_like(PP_T_12(s)), PP_T_12(s))
	return matrix_element
