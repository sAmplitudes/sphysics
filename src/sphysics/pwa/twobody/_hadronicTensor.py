# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Skeleton for two-body hadronic tensors
'''

# pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division

from typing import Any

import numpy as np

from ... import lorentz
from ... import math
from ...utils import Logger
from ._kinematics import Kinematics


log = Logger('hadCurr')


class HadronicTensor(Kinematics):
	'''
	Skeleton container for two-body hadronic tensor definitions.
	'''

	def __init__(self, p1: np.ndarray = None, p2: np.ndarray = None) -> None:
		self._g12 = None
		self._q12 = None
		self._k12 = None
		super().__init__(p1, p2)

	def reset(self) -> None:
		'''
		Reset cached event-level quantities.
		'''
		super().reset()
		self._g12 = None
		self._q12 = None
		self._k12 = None

	@property
	def g12(self) -> np.ndarray:
		''' Projector of the transversal component to the (12) system'''
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
	def wave_1m(self) -> np.ndarray:
		''' Current for a P-wave in the (12) system, .ie., :math:`1^- \\to [(1) (2)]_P` '''
		return self.k12

	@property
	def wave_0p(self) -> np.ndarray:
		''' Current for an S-wave in the (12) system, .ie., :math:`0^+ \\to [(1) (2)]_S` '''
		return self.p12

	def __getitem__(self, tensorName: str) -> Any:
		'''
		Access a tensor by its attribute name.
		'''
		return getattr(self, tensorName)


class Wave:
	"""Class representing a two-body partial wave, i.e. the quantum numbers of the decaying system.

	String label format: ``{J}{P}``, e.g.
		- ``1m`` for :math:`J^P = 1^-`
		- ``0p`` for :math:`J^P = 0^+`

	These map directly to the ``wave_1m`` / ``wave_0p`` tensor properties on :class:`HadronicTensor`.
	"""
	chargeLabel = {'m': -1, 'p': +1}

	def __init__(self, J_X: int, P_X: int) -> None:
		self.J_X = int(J_X)
		self.P_X = self.chargeLabel[P_X] if isinstance(P_X, str) else int(P_X)

		if self.P_X not in (-1, 1):
			log.raiseException(ValueError, f'P_X of {self.P_X} is not allowed')

	@classmethod
	def fromString(cls, name: str) -> 'Wave':
		"""Construct a :class:`Wave` from its string label, e.g. ``'wave_1m'`` or ``'1m'``."""
		name = name.lstrip('wave_')
		if len(name) != 2:
			log.raiseException(ValueError, f'Cannot parse wave name "{name}": expected format "<J><P>", e.g. "1m"')
		J_X, P_X = list(name)
		return cls(J_X, P_X)

	def __repr__(self) -> str:
		return f'{self.J_X}{"p" if self.P_X == +1 else "m"}'

	def __str__(self) -> str:
		return self.__repr__()
