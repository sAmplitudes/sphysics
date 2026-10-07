# coding: utf-8
# Python reimplementation of ROOTPWA's piPiSWaveAuMorganPenningtonKachaev
'''
Created on Friday 12 01 2024
Author: Godo Kurten
Description: 	Amplitude for [ππ]S wave from Au Morgan Pennington.
				[K.L. Au et al, Phys. Rev. D35, 1633]
				############################################################
				Kachaev's version of the AMP pi+ pi- s-wave parameterization

				from the original fortran code in fortran/eps_mx.f:
				[K.L. Au et al, Phys. Rev. D35, 1633]

				original comments:
				04-Mar-2003 See fortran/eps_k1.f for description.
				Here matrix M = K^{-1} is parametrized with one pole.
				Misprint in the article (other than in K1--K3 solutions)
				was corrected.

				14-Mar-2003 Nice amplitude for pi-pi S-wave without f0(975).
				It is smooth and nicely tends to zero after approx 1.5 GeV.
				f0(975) pole excluded; coupling to KK zeroed; set C411=C422=0.
				The largest effect from C411, zeroing of C422 looks insignificant.
'''
#pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import tensorflow as tf

from .._constants import Constants
from ..kinematics._kinematics import twobodyBreakupmomentum
from .. import math
from ._utils import prepare_mass_tensor

mPin = Constants.M.pi0 #neutral pi mass
mPic = Constants.M.pi  #charged pi mass
mKn = Constants.M.K0   #neutral K mass
mKc = Constants.M.K    #charged K mass
mKmean = (mKn + mKc)/2

def piPiSwaveAuMorganPennington(mass):
	"""
	Amplitude for [ππ]S wave from Au Morgan Pennington.
	[K.L. Au et al, Phys. Rev. D35, 1633]

	:param mass: Can be either `tf.Tensor` or `np.ndarray`.
	:returns: If `mass` consists of a single value, the amplitude is given for that value.
			If `mass` consists of :math:`n` values, :math:`n` amplitude values are returned.
	"""
	mass = prepare_mass_tensor(mass)
	SWave = piPiSWaveAuMorganPenningtonImpl()
	SWave.adjust_type_attribute(mass)
	rho = SWave.calculateRho(mass)
	massMmatrix = SWave.calculateM(mass)
	massAmp = SWave.calculateT(rho, massMmatrix)[:,0,0]
	return massAmp

def piPiSwaveAuMorganPenningtonKachaev(mass):
	"""
	Kachaev's version of the AMP :math:`\\pi^+ \\pi^-` S-wave parameterization.
	from the original fortran code in fortran/eps_mx.f:
	[K.L. Au et al, Phys. Rev. D35, 1633]

	:param mass: Can be either `tf.Tensor` or `np.ndarray`.
	:returns: If `mass` consists of a single value, the amplitude is given for that value.
			If `mass` consists of :math:`n` values, :math:`n` amplitude values are returned.
	"""
	mass = prepare_mass_tensor(mass)
	SWave = piPiSwaveAuMorganPenningtonKachaevImpl()
	SWave.adjust_type_attribute(mass)
	rho = SWave.calculateRho(mass)
	massMmatrix = SWave.calculateM(mass)
	massAmp = SWave.calculateT(rho, massMmatrix)[:,0,0]
	return massAmp

class piPiSWaveAuMorganPenningtonImpl:
	'''
	This class implements the basic [ππ]S wave of Au Morgan and Pennington.
	The amplitude can be calculated with the function calculateAmplitude.
	'''

	def __init__(self):
		self.a = np.empty([2,2,2],dtype=complex)
		self.c = np.empty([5,2,2],dtype=complex)
		self.sp = np.empty([2])
		self.f = np.empty([2])
		self.set_parameters()

	def set_parameters(self):
		'''
		This function sets the necessary constants for the calculation of the amplitude
		'''
		self.f[0] = 0.1968   #AMP Table 1, M solution: f_1^1 and f_2^1
		self.f[1] = -0.0154  #AMP Table 1, M solution: f_1^1 and f_2^1

		self.a[0,0,0] =  0.1131  # AMP Table 1, M solution: f_2^2
		self.a[0,0,1] =  0.0150  # AMP Table 1, M solution: f_1^3
		self.a[0,1,0] =  0.0150  # AMP Table 1, M solution: f_1^3
		self.a[0,1,1] = -0.3216  # AMP Table 1, M solution: f_2^3
		self.a[1,0,0] = self.f[0] * self.f[0]
		self.a[1,0,1] = self.f[0] * self.f[1]
		self.a[1,1,0] = self.f[1] * self.f[0]
		self.a[1,1,1] = self.f[1] * self.f[1]

		self.c[0,0,0] =  0.0337                # AMP Table 1, M solution: c_11^0
		self.c[1,0,0] = -0.3185                # AMP Table 1, M solution: c_11^1
		self.c[2,0,0] = -0.0942                # AMP Table 1, M solution: c_11^2
		self.c[3,0,0] = -0.5927                # AMP Table 1, M solution: c_11^3
		self.c[4,0,0] =  0.1957                # AMP Table 1, M solution: c_11^4
		self.c[0,0,1] = self.c[0,1,0] = -0.2826  # AMP Table 1, M solution: c_12^0
		self.c[1,0,1] = self.c[1,1,0] =  0.0918  # AMP Table 1, M solution: c_12^1
		self.c[2,0,1] = self.c[2,1,0] =  0.1669  # AMP Table 1, M solution: c_12^2
		self.c[3,0,1] = self.c[3,1,0] = -0.2082  # AMP Table 1, M solution: c_12^3
		self.c[4,0,1] = self.c[4,1,0] = -0.1386  # AMP Table 1, M solution: c_12^4
		self.c[0,1,1] =  0.3010                # AMP Table 1, M solution: c_22^0
		self.c[1,1,1] = -0.5140                # AMP Table 1, M solution: c_22^1
		self.c[2,1,1] =  0.1176                # AMP Table 1, M solution: c_22^2
		self.c[3,1,1] =  0.5204                # AMP Table 1, M solution: c_22^3
		self.c[4,1,1] = -0.3977                # AMP Table 1, M solution: c_22^4

		self.sp[0] = -0.0074  # AMP Table 1, M solution: s_0
		self.sp[1] =  0.9828  # AMP Table 1, M solution: s_1

	def adjust_type_attribute(self, mass):
		if tf.is_tensor(mass):
			self.f = tf.convert_to_tensor(self.f)
			self.a = tf.convert_to_tensor(self.a)
			self.c = tf.convert_to_tensor(self.c)
			self.sp = math.castToComplex(tf.convert_to_tensor(self.sp))

	def calculateRho(self, mass):
		'''
		This function takes mass as a variable and calculates the phase-space matrix
		@Param: mass can either be tf.Tensor or np.ndarray
		@return: if mass consists of a single value a 2x2 matrix is returned
		         if mass consists of n values a nx2x2 matrix is returned. The first dimension corresponds to the different mass values
		'''
		qPicPic = twobodyBreakupmomentum(math.castToComplex(mass**2), mPic, mPic)
		qPinPin = twobodyBreakupmomentum(math.castToComplex(mass**2), mPin, mPin)
		qKcKc = twobodyBreakupmomentum(math.castToComplex(mass**2), mKc, mKc)
		qKnKn = twobodyBreakupmomentum(math.castToComplex(mass**2), mKn, mKn)
		if tf.is_tensor(mass):
			# prefactor_00 = [1,0,1,0,....] so that every second element is set to zero
			prefactor_00 = math.castToComplex(1 - tf.math.mod(tf.range(len(mass)*2), 2))
			# prefactor_00 = [0,1,0,1,....] so that every second element is set to zero
			prefactor_11 = math.castToComplex(tf.math.mod(tf.range(len(mass)*2), 2))
			# element_00 is tensor, where every entry appears twice. With prefactor only the desired elements have non-zero value
			element_00 = tf.repeat((((2. * qPicPic) / mass + (2. * qPinPin) / mass) / 2.),repeats=2)*prefactor_00
			element_11 = tf.repeat((((2. * qKcKc)   / mass + (2. * qKnKn)   / mass) / 2.),repeats=2)*prefactor_11
			massrho = tf.reshape(tf.stack([element_00,element_11],axis=1), (len(mass),2,2))
		else:
			massrho = math.castToComplex(math.zeros([len(mass),2,2], mass))
			massrho[:,0,0] = ((2. * qPicPic) / mass + (2. * qPinPin) / mass) / 2.
			massrho[:,1,1] = ((2. * qKcKc)   / mass + (2. * qKnKn)   / mass) / 2.
		return massrho

	def calculateM(self, mass):
		'''
		This function takes mass as a variable and calculates the M matrix
		@Param: mass can either be tf.Tensor or np.npdarray
		@return: if mass consists of a single value a 2x2 matrix is returned
		         if mass consists of n values a nx2x2 matrix is returned. The first dimension corresponds to the different mass values
		'''
		fa = 1. / (math.pow(math.expand_dims(mass, 1), 2) - self.sp)
		scale = (math.pow(mass, 2) / (4 * mKmean * mKmean)) - 1
		sc = math.pow(math.expand_dims(scale,1), math.castToComplex(math.arange(0,5,fa))) #This takes scale makes a vector of it and raises it to the power of the element
		massMmatrix = math.tensordot(fa, self.a, axes=(1,0)) + math.tensordot(sc, self.c, axes=(1,0))
		return massMmatrix

	def calculateT(self, rho, massMmatrix):
		'''
		This function takes mass and kachaev as a variable and calculates the T matrix
		@Param: mass can either be tf.Tensor or np.npdarray
		        kachaev: if it is assigned a value the [0,1] and [1,0] elements of the M matrix are set to zero
		@return: if mass consists of a single value a 2x2 matrix is returned
		         if mass consists of n values a nx2x2 matrix is returned. The first dimension corresponds to the different mass values
		'''
		massTmatrix = math.invert(massMmatrix - 1j * rho)
		return massTmatrix


class piPiSwaveAuMorganPenningtonKachaevImpl(piPiSWaveAuMorganPenningtonImpl):
	'''
	This class corresponds to Kachaevs version of the AuMorganPennington [ππ]S amplitude
	This amplitude can be calculated by calculateAmplitude
	'''

	def __init__(self):
		super().__init__()
		self.c[4,0,0] = 0
		self.c[4,1,1] = 0

		self.a[0,0,1] = 0
		self.a[0,1,0] = 0

		self.a[1,0,0] = 0
		self.a[1,0,1] = 0
		self.a[1,1,0] = 0
		self.a[1,1,1] = 0

	def calculateM(self, mass):
		'''
		This function takes mass as a variable and calculates the M matrix
		@Param: mass can either be tf.Tensor or np.npdarray
		@return: if mass consists of a single value a 2x2 matrix is returned
		         if mass consists of n values a nx2x2 matrix is returned. The first dimension corresponds to the different mass values
		'''
		massMmatrix = super().calculateM(mass)
		# Change M matrix entries to match Kachaevs version of the amplitude, i.e. set elements [0,1] and [1,0] of M matrix to zero
		if tf.is_tensor(mass):
			# Create list with indices that are supposed to be 0
			indices = tf.constant([[i, 0, 1] for i in range(massMmatrix.shape[0])] + [[i, 1, 0] for i in range(massMmatrix.shape[0])])
			# Create updates with zeros
			updates = tf.zeros(indices.shape[0], dtype=massMmatrix.dtype)
			# Use tf.tensor_scatter_nd_update to set selected elements to zero
			massMmatrix = tf.tensor_scatter_nd_update(massMmatrix, indices, updates)
		else:
			# Set certain elements to zero according to Kachaevs swave amplitude
			massMmatrix[:,0,1] = 0
			massMmatrix[:,1,0] = 0
		return massMmatrix
