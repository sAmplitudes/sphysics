# coding: utf-8
'''
:Author: Stefan Wallner
:description: Class calculating various three-body hadronic currents, Created on Tuesday 15 03 2022
'''

# pylint: disable=invalid-name,attribute-defined-outside-init,too-many-public-methods

from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from ... import lorentz
from ... import math
from ... import _constants
from ...utils import Logger
from ._kinematics import Kinematics

log = Logger('hadCurr')


class HadronicTensors(Kinematics):
	'''
	Class for calculating various three-body hadronic currents

	The isobar system is always the (12) system.
	'''

	def reset(self) -> None:
		super().reset()

		self._g12 = None
		self._q12 = None
		self._k12 = None

		self._g123 = None
		self._q123 = None
		self._k123 = None

		self._tensor_12_Swave = None
		self._tensor_12_Pwave = None
		self._tensor_12_Dwave = None
		self._tensor_12_Fwave = None

		self._tensor_123_Swave = None
		self._tensor_123_Pwave = None
		self._tensor_123_Dwave = None
		self._tensor_123_Fwave = None


	@property
	def g12(self) -> np.ndarray:
		''' Projector of the transversal component to the (12) sub-system'''
		if self._g12 is None:
			self._g12 = lorentz.eta[:,:,None] - lorentz.tensorproduct(self.p12, self.p12)/self.s12
		return self._g12

	@property
	def q12(self) -> np.ndarray:
		''' Difference between the 1 and 2 momenta, e.g. in the (12) rest frame equals to the two-body break-up momentum of the (12) :math:`\\to` 1 2 decay'''
		if self._q12 is None:
			self._q12 = 0.5*(self.p1 - self.p2)
		return self._q12

	@property
	def k12(self) -> np.ndarray:
		''' 4-momentum transversal to p12 '''
		if self._k12 is None:
			self._k12 = math.einsum('ije,j,je->ie', self.g12, lorentz.et, self.q12)
		return self._k12


	@property
	def g123(self) -> np.ndarray:
		''' Projector of the transversal component to the (123) system'''
		if self._g123 is None:
			self._g123 = lorentz.eta[:,:,None] - lorentz.tensorproduct(self.p123, self.p123)/self.s123
		return self._g123

	@property
	def q123(self) -> np.ndarray:
		''' Difference between the (12) and 3 momenta, e.g. in the (123) rest frame equals to the two-body break-up momentum of the (123) :math:`\\to` (12) 3 decay'''
		if self._q123 is None:
			self._q123 = 0.5*(self.p12 - self.p3)
		return self._q123

	@property
	def k123(self) -> np.ndarray:
		''' 4-momentum transversal to p123 '''
		if self._k123 is None:
			self._k123 = math.einsum('ije,j,je->ie', self.g123, lorentz.et, self.q123)
		return self._k123


	@property
	def tensor_12_Swave(self) -> np.ndarray:
		''' Tensor representing the (12) sub-system being in an S-wave '''
		if self._tensor_12_Swave is None:
			self._tensor_12_Swave = 1.
		return self._tensor_12_Swave

	@property
	def tensor_12_Pwave(self) -> np.ndarray:
		''' Tensor representing the (12) sub-system being in an P-wave '''
		if self._tensor_12_Pwave is None:
			self._tensor_12_Pwave = self.k12
		return self._tensor_12_Pwave

	@property
	def tensor_12_Dwave(self) -> np.ndarray:
		''' Tensor representing the (12) sub-system being in an D-wave '''
		if self._tensor_12_Dwave is None:
			self._tensor_12_Dwave = 0.5*( 3*lorentz.tensorproduct(self.k12, self.k12)  - self.g12*lorentz.lp(self.k12,self.k12))
		return self._tensor_12_Dwave

	@property
	def tensor_12_Fwave(self) -> np.ndarray:
		''' Tensor representing the (12) sub-system being in an F-wave '''
		if self._tensor_12_Fwave is None:
			self._tensor_12_Fwave = 0.5*( 5.*lorentz.tensorproduct(self.k12, self.k12, self.k12)
			                             - lorentz.lp(self.k12,self.k12)*(  lorentz.tensorproduct(self.k12, self.g12)
										                                  + math.transpose(lorentz.tensorproduct(self.k12, self.g12), (1,0,2,3))
										                                  + lorentz.tensorproduct(self.g12, self.k12)
																		  )
		                                )
		return self._tensor_12_Fwave


	@property
	def tensor_123_Swave(self) -> np.ndarray:
		''' Tensor representing the (123) system being in an S-wave '''
		if self._tensor_123_Swave is None:
			self._tensor_123_Swave = 1.
		return self._tensor_123_Swave

	@property
	def tensor_123_Pwave(self) -> np.ndarray:
		''' Tensor representing the (123) system being in an P-wave '''
		if self._tensor_123_Pwave is None:
			self._tensor_123_Pwave = self.k123
		return self._tensor_123_Pwave

	@property
	def tensor_123_Dwave(self) -> np.ndarray:
		''' Tensor representing the (123) system being in an D-wave '''
		if self._tensor_123_Dwave is None:
			self._tensor_123_Dwave = 0.5*( 3*lorentz.tensorproduct(self.k123, self.k123)  - self.g123*lorentz.lp(self.k123,self.k123))
		return self._tensor_123_Dwave

	@property
	def tensor_123_Fwave(self) -> np.ndarray:
		''' Tensor representing the (123) system being in an F-wave '''
		if self._tensor_123_Fwave is None:
			self._tensor_123_Fwave = 0.5*( 5.*lorentz.tensorproduct(self.k123, self.k123, self.k123)
			                             - lorentz.lp(self.k123,self.k123)*(  lorentz.tensorproduct(self.k123, self.g123)
										                                    + math.transpose(lorentz.tensorproduct(self.k123, self.g123), (1,0,2,3))
										                                    + lorentz.tensorproduct(self.g123, self.k123)
																		   )
		                                )
		return self._tensor_123_Fwave


	@property
	def wave_1p_0p_P(self) -> np.ndarray:
		''' Current for wave :math:`1^+ \\to [(12)_0^+ (3) ]_P` '''
		return self.tensor_123_Pwave


	@property
	def wave_1p_1m_S(self) -> np.ndarray:
		''' Current for wave :math:`1^+ \\to [ (12)_1^- (3) ]_S` '''
		return math.einsum('mne,n,ne->me', self.g123, lorentz.et, self.tensor_12_Pwave)


	@property
	def wave_1p_1m_D(self) -> np.ndarray:
		''' Current for wave :math:`1^+ \\to [ (12)_1^- (3) ]_D` '''
		return math.einsum('mne,n,ne->me', self.tensor_123_Dwave, lorentz.et, self.tensor_12_Pwave)


	@property
	def wave_1p_2p_P(self) -> np.ndarray:
		''' Current for wave :math:`1^+ \\to [ (12)_2^+ (3) ]_D` '''
		return math.einsum('mne,n,r,nre,re->me', self.g123, lorentz.et, lorentz.et, self.tensor_12_Dwave, self.tensor_123_Pwave)

	@property
	def wave_1p_2p_F(self) -> np.ndarray:
		''' Current for wave :math:`1^+ \\to [ (12)_2^+ (3) ]_F` '''
		return math.einsum('n,r,nre,mnre->me', lorentz.et, lorentz.et, self.tensor_12_Dwave, self.tensor_123_Fwave)

	@property
	def wave_1p_3m_D(self) -> np.ndarray:
		''' Current for wave :math:`1^+ \\to [ (12)_3^- (3) ]_D` '''
		return math.einsum('n,r,s,mne,nrse,rse->me', lorentz.et, lorentz.et, lorentz.et, self.g123, self.tensor_12_Fwave, self.tensor_123_Dwave)

	@property
	def wave_1m_1m_P(self) -> np.ndarray:
		''' Current for wave :math:`1^- \\to [ (12)_1^- (3) ]_P` '''
		return math.einsum('mnrs,n,ne,r,re,s,se->me',
		                    lorentz.levitCivitaSymbol, lorentz.et, self.p123, lorentz.et, self.tensor_123_Pwave, lorentz.et, self.tensor_12_Pwave)


	@property
	def wave_0m_0p_S(self) -> np.ndarray:
		''' Current for wave :math:`0^- \\to [ (12)_0^+ (3) ]_S` '''
		return self.p123


	@property
	def wave_0m_1m_P(self) -> np.ndarray:
		''' Current for wave :math:`0^- \\to [ (12)_1^- (3) ]_P` '''
		return self.p123*math.einsum('ne,n,ne->e', self.tensor_123_Pwave, lorentz.et, self.tensor_12_Pwave)


	@property
	def wave_0m_2p_D(self) -> np.ndarray:
		''' Current for wave :math:`0^- \\to [ (12)_2^+ (3) ]_D` '''
		return self.p123*math.einsum('nre,n,r,nre->e', self.tensor_123_Dwave, lorentz.et, lorentz.et, self.tensor_12_Dwave)


	@property
	def wave_0m_3m_F(self) -> np.ndarray:
		'''Current for wave :math:`0^- \\to [ (12)_3^- (3) ]_F` '''
		return self.p123*math.einsum('nrse,n,r,s,nrse->e', self.tensor_123_Fwave, lorentz.et, lorentz.et, lorentz.et, self.tensor_12_Fwave)


class Wave:
	"""Class representing a 3-body partial wave, i.e. a set of quantum numbers describing the state of three pseudoscalar particles

	String label formats are e.g.
		- 1p_1m_S
		- 1p_rho(770)_S
		- 1p_rho(770)[1m]_S
	"""
	chargeLabel = {'m': -1, 'p': +1}

	def __init__(self, J_X: int, P_X: int, J_isobar: int, P_isobar: int, isobarLabel: str, L: int) -> None:
		self.J_X = int(J_X)
		self.P_X = self.chargeLabel[P_X] if isinstance(P_X, str) else int(P_X)
		self.J_isobar = int(J_isobar)
		self.P_isobar = self.chargeLabel[P_isobar] if isinstance(P_isobar, str) else int(P_isobar)
		self.isobarLabel = isobarLabel
		self.L = _constants.Constants.spectroscopyMomentumNotation[L] if isinstance(L, str) else int(L)

		if self.P_X not in (-1,1):
			log.raiseException(ValueError, f"P_X of {self.P_X} is not allowed")
		if self.P_isobar not in (-1,1):
			log.raiseException(ValueError, f"P_isobar of {self.P_isobar} is not allowed")


	@classmethod
	def fromString(cls, name: str)-> Wave:

		X, isobar, L = name.lstrip('wave_').split('_')
		J_X, P_X = list(X)
		if '[' in isobar:
			isobarLabel, isobarQN = isobar.split(']')
			J_isobar, P_isobar, *_ = list(isobarQN)
		elif len(isobar) == 2:
			J_isobar, P_isobar = list(isobar)
			isobarLabel = None
		else:
			isobarLabel = isobar
			J_isobar = P_isobar = None
		return cls(J_X, P_X, J_isobar, P_isobar, isobarLabel, L)


	def __repr__(self) -> str:
		if self.isobarLabel is None:
			isobar = f'{self.J_isobar}{"p" if self.P_isobar == +1 else "m"}'
		else:
			isobar = f'{self.isobarLabel}[{self.J_isobar}{"p" if self.P_isobar == +1 else "m"}]'
		return f'{self.J_X}{"p" if self.P_X == +1 else "m"}_{isobar}_{_constants.Constants.spectroscopyMomentumNotation[self.L]}'


	def __str__(self) -> str:
		return self.__repr__()
