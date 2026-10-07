# coding: utf-8
'''
Created on Thursday 03 11 2022
Author: Stefan Wallner
Description:  Classes for amplitude calculation of dalitz-plot analyses
'''

# pylint: disable=invalid-name,attribute-defined-outside-init,too-many-public-methods

from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from ._kinematics import Kinematics, cosTheta_12
from ...utils import Logger
from ... import math

log = Logger('dalitzPWA')





class DalitzAmplitudes3P_Zemach(Kinematics):
	'''
	Class for calculating partial wave amplitudes for dalitz-plot analysis of

	.. math::
		P \\to P_1 P_2 P_3

	where P is a pseudo-scalar state.

	The Zemach formalism is used to calculate the partial-wave amplitudes.
	See eq. 13.2.7 and eq. 1.2.6 with :math:`m_r` replaced by :math:`m_{ab}` of `The Physics of B Factories
	<https://link.springer.com/article/10.1140/epjc/s10052-014-3026-9#preview>`_ 74:3026.

	The isobar system is always the (12) system.

	The spin-analyzer is is :math:`P_1` in the (12) rest frame,
	i.e. :math:`q=p_1^*`, where :math:`p_1^*` is :math:`p_1` in the (12) rest frame,
	in equation 13.2.7 in the reference above.
	'''

	@property
	def wave_0m_0p_S(self) -> np.ndarray:
		''' Amplitude for wave :math:`0^- \\to [ (12)_0^+ (3) ]_S` '''
		return 1.0


	@property
	def wave_0m_1m_P(self) -> np.ndarray:
		''' Amplitude for wave :math:`0^- \\to [ (12)_1^- (3) ]_P` '''
		return self.m13**2 - self.m23**2 - (self.m123**2-self.m3**2)*(self.m1**2-self.m2**2)/self.m12**2


	@property
	def wave_0m_2p_D(self) -> np.ndarray:
		''' Amplitude for wave :math:`0^- \\to [ (12)_2^+ (3) ]_D` '''
		return     ( self.m23**2-self.m13**2 + (self.m123**2-self.m3**2)*(self.m1**2-self.m2**2)/self.m12**2  )**2 \
		    -1./3.*( self.m12**2-2.*self.m123**2-2.*self.m3**2 + (self.m123**2-self.m3**2)**2/self.m12**2     )    \
				  *( self.m12**2-2.*self.m1**2-2.*self.m2**2 + (self.m1**2-self.m2**2)**2/self.m12**2         )

	@property
	def wave_0m_3m_F(self) -> np.ndarray:
		'''
		Amplitude for wave :math:`0^- \\to [ (12)_3^- (3) ]_F`
		Zeemach tensor form from `LHCb amplitude analysis paper <https://cds.cern.ch/record/2205214/files/LHCB-PAPER-2016-026.pdf>`_.
		Translation to mass dep. term from `The Physics of B Factories <https://arxiv.org/pdf/1406.6311>`_ on page 169.
		'''
		abs_pq_sq = 1/2*(( self.m12**2-2.*self.m123**2-2.*self.m3**2 + (self.m123**2-self.m3**2)**2/self.m12**2)   \
						*( self.m12**2-2.*self.m1**2-2.*self.m2**2 + (self.m1**2-self.m2**2)**2/self.m12**2))
		cosTheta = cosTheta_12(self.m1, self.m2, self.m3, self.m12, self.m13, self.m123)
		return -(24/15)*(math.sqrt(abs_pq_sq)**3)*(5*cosTheta**3-3*cosTheta)

	@property
	def wave_0m_4p_G(self) -> np.ndarray:
		'''
		Amplitude for wave :math:`0^- \\to [ (12)_4^+ (3) ]_G`
		Zeemach tensor form from `Wigner D-Matrix <https://en.wikipedia.org/wiki/Wigner_D-matrix>`_ and
		`Jacobi Polynomials <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_ with :math:`j=4, m'=m=0`.
		Prefactor is an educated guess. Prefactor does not matter, since wave is normalized.
		Prefactors of previous waves follow

		.. math::
			(-1)^l \\cdot \\frac{2^l \\cdot l!}{\\prod_{i=0}^{l} (2i - 1)}

		This system is continued for the G-wave and altered by the prefactor from Legendre polynomials (like above).
		'''
		abs_pq_sq = 1/2*(( self.m12**2-2.*self.m123**2-2.*self.m3**2 + (self.m123**2-self.m3**2)**2/self.m12**2)   \
						*( self.m12**2-2.*self.m1**2-2.*self.m2**2 + (self.m1**2-self.m2**2)**2/self.m12**2))
		cosTheta = cosTheta_12(self.m1, self.m2, self.m3, self.m12, self.m13, self.m123)
		return +(16/35)*(abs_pq_sq**2)*(35*cosTheta**4-30*cosTheta**2+3)
