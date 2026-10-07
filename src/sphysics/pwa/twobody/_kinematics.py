# coding: utf-8
'''
Created on Thursday 03 11 2022
Author: Stefan Wallner
Description: Class that calculates two-body kinematic variables from 4-momenta
'''

# pylint: disable=invalid-name,attribute-defined-outside-init,too-many-public-methods

from __future__ import absolute_import, print_function, division, annotations

from typing import Optional

import numpy as np

from ... import lorentz
from ... import math
from ...utils import Logger


log = Logger('hadCurr')


class Kinematics(object):
	'''
	Class for calculating various two-body kinematics.
	'''

	def __init__(self, p1: Optional[np.ndarray] = None, p2: Optional[np.ndarray] = None, momentaInCMS: bool = False) -> None:
		self._momentaInCMS = momentaInCMS
		if p1 is not None:
			if p2 is None:
				log.raiseException(ValueError, 'p2 must be set when p1 is set')
			self.setEvents(p1, p2)
		else:
			self.reset()

	def reset(self) -> None:
		'''
		Reset all cached kinematic variables.
		'''
		self._p1 = None
		self._p2 = None

		self._m1 = None
		self._m2 = None

		self._p12 = None
		self._s12 = None
		self._m12 = None

	def __getstate__(self):
		self.reset()  # clear data before pickling this object.
		return self.__dict__.copy()

	def __setstate__(self, state):
		self.__dict__.update(state)

	def __getitem__(self, name):
		return getattr(self, name)

	def setEvents(self, p1: np.ndarray, p2: np.ndarray) -> None:
		'''
		Set event four-momenta and reset all cached calculated quantities.
		'''
		if p2.shape != p1.shape:
			log.raiseException(ValueError, 'The 4-momenta must have the same shape!')
		if len(p1.shape) != 2 or p1.shape[0] != 4:
			log.raiseException(ValueError, 'The 4-momenta must have the shape (4,nEvents), but have the shape {0}!'.format(p1.shape))

		self.reset()
		self._p1 = p1
		self._p2 = p2

	def setMasses(self, m12: np.ndarray,
	              m1: np.ndarray,
	              m2: np.ndarray) -> None:
		'''
		Set masses for kinematic calculations without full 4-momenta.
		'''
		self.reset()
		self._m12 = m12
		self._m1 = m1
		self._m2 = m2

	@property
	def momentaInCMS(self) -> bool:
		return self._momentaInCMS

	@property
	def p1(self) -> np.ndarray:
		if self._p1 is None:
			log.raiseException(Exception, '`Kinematics` object has no events set')
		return self._p1

	@property
	def p2(self) -> np.ndarray:
		if self._p2 is None:
			log.raiseException(Exception, '`Kinematics` object has no events set')
		return self._p2

	@property
	def p12(self) -> np.ndarray:
		if self._p12 is None:
			self._p12 = self.p1 + self.p2
		return self._p12

	@property
	def s12(self) -> np.ndarray:
		if self._s12 is None:
			self._s12 = lorentz.lp(self.p12, self.p12)
		return self._s12

	@property
	def m1(self):
		if self._m1 is None:
			self._m1 = math.sqrt(lorentz.lp(self.p1, self.p1))
		return self._m1

	@property
	def m2(self):
		if self._m2 is None:
			self._m2 = math.sqrt(lorentz.lp(self.p2, self.p2))
		return self._m2

	@property
	def m12(self):
		if self._m12 is None:
			self._m12 = math.sqrt(self.s12)
		return self._m12
