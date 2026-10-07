# coding: utf-8
'''
Created on Thursday 03 11 2022
Author: Stefan Wallner
Description: Class that calculates kinematic variables from 4-momenta
'''

# pylint: disable=invalid-name,attribute-defined-outside-init,too-many-public-methods

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import tensorflow as tf

from ... import lorentz
from ... import math
from ...utils import Logger
from ..utils import calculateIsobarTreeHelicityAngles


log = Logger('hadCurr')

class Kinematics(object):
	'''
	Class for calculating various three-body kinematics

	'''
	def __init__(self, p1: np.ndarray = None, p2: np.ndarray = None ,p3: np.ndarray = None,
			  momentaInCMS: bool = False) -> None:
		"""summary

		Args:
			p1 (np.ndarray, optional): 4-momentum of particle 1 of the shape ((E, px, py, px), events).
			p2 (np.ndarray, optional): 4-momentum of particle 2 of the shape ((E, px, py, px), events).
			p3 (np.ndarray, optional): 4-momentum of particle 3 of the shape ((E, px, py, px), events).
		    momentaInCMS (bool, optional):  True if the momenta are given in their CMS. In this case, the helicity angles of the
		                                    decay of the full (1,2,3) system cannot be calculated and the z axis is used as
		    								reference plane of the subsystem decays.
		"""
		self._momentaInCMS = momentaInCMS
		if p1 is not None:
			self.setEvents(p1, p2, p3)
		else:
			self.reset()

	def reset(self) -> None:
		'''Resets all kinematic variables.
		'''
		self._p1 = None
		self._p2 = None
		self._p3 = None

		self._boost_12RF = None
		self._p1_12RF = None
		self._p2_12RF = None
		self._p3_12RF = None

		self._m1 = None
		self._m2 = None
		self._m3 = None

		self._p12 = None
		self._p13 = None
		self._p23 = None
		self._p123 = None
		self._s12 = None
		self._s13 = None
		self._s23 = None
		self._s123 = None
		self._m12 = None
		self._m13 = None
		self._m23 = None
		self._m123 = None

		self._helicityAngles_12 = None
		self._helicityAngles_13 = None

	def __getstate__(self):
		self.reset() # clear data before pickling this object.
		return self.__dict__.copy()

	def __setstate__(self, state):
		self.__dict__.update(state)


	def __getitem__(self, name):
		return getattr(self, name)


	def setEvents(self, p1: float|np.ndarray|tf.Tensor, p2: float|np.ndarray|tf.Tensor, p3: float|np.ndarray|tf.Tensor) -> None:
		"""Set the event four momenta and resets all cached calculated quantities.

		Args:
			p1 (float|np.ndarray|tf.Tensor): 4-momenta of the first particle of the shape (4, events)
			p2 (float|np.ndarray|tf.Tensor): 4-momenta of the second particle of the shape (4, events)
			p3 (float|np.ndarray|tf.Tensor): 4-momenta of the third particle of the shape (4, events)

		Raises:
			ValueError: If the 4-momenta do not have the same shape or if the shape is not (4, nEvents)

		4-momenta shape:
			shape ((E, px, py, px), events)
		"""
		if p2.shape != p1.shape or p3.shape != p1.shape:
			log.raiseException(ValueError, "The 4-momenta must have the same shape!")
		if len(p1.shape) != 2 or p1.shape[0] != 4:
			log.raiseException(ValueError, "The 4-momenta must have the shape (4,nEvents), but have the shape {0}!".format(p1.shape))

		self.reset()

		self._p1 = p1
		self._p2 = p2
		self._p3 = p3


	def setMasses(self, m123: float|np.ndarray|tf.Tensor,
	              m12: float|np.ndarray|tf.Tensor, m13: float|np.ndarray|tf.Tensor, m23: float|np.ndarray|tf.Tensor,
				  m1: float|np.ndarray|tf.Tensor, m2: float|np.ndarray|tf.Tensor, m3: float|np.ndarray|tf.Tensor) -> None:
		"""Set the masses for the kinematic calculations.

		Args:
			m123 (float|np.ndarray|tf.Tensor): Mass of the parent particle or system.
			m12 (float|np.ndarray|tf.Tensor): Mass of the (1,2) subsystem.
			m13 (float|np.ndarray|tf.Tensor): Mass of the (1,3) subsystem.
			m23 (float|np.ndarray|tf.Tensor): Mass of the (2,3) subsystem.
			m1 (float|np.ndarray|tf.Tensor): Mass of particle 1.
			m2 (float|np.ndarray|tf.Tensor): Mass of particle 2.
			m3 (float|np.ndarray|tf.Tensor): Mass of particle 3.

		This method sets the masses for all particles and subsystems, allowing for
		calculations of kinematic variables without needing the full 4-momenta.
		It resets all cached calculations to ensure consistency with the new masses.
		"""

		self.reset()

		self._m123 = m123
		self._m12 = m12
		self._m13 = m13
		self._m23 = m23
		self._m1 = m1
		self._m2 = m2
		self._m3 = m3


	@property
	def boost_12RF(self) -> np.ndarray:
		'''Momentum boosted into rest frame of :math:`p_{12}`
		'''
		if self._boost_12RF is None:
			self._boost_12RF = lorentz.getBoostToRestFrame(self.p12)
		return self._boost_12RF

	@property
	def p1_12RF(self) -> np.ndarray:
		'''Momentum :math:`p_1` boosted in the reference frame of :math:`p_{12}` '''
		if self._p1_12RF is None:
			self._p1_12RF = math.einsum('ije,je->ie', self.boost_12RF, self.p1)
		return self._p1_12RF

	@property
	def p2_12RF(self) -> np.ndarray:
		'''Momentum :math:`p_2` boosted in the reference frame of :math:`p_{12}` '''

		if self._p2_12RF is None:
			self._p2_12RF = math.einsum('ije,je->ie', self.boost_12RF, self.p2)
		return self._p2_12RF

	@property
	def p3_12RF(self) -> np.ndarray:
		'''Momentum :math:`p_3` boosted in the reference frame of :math:`p_{12}` '''

		if self._p3_12RF is None:
			self._p3_12RF = math.einsum('ije,je->ie', self.boost_12RF, self.p3)
		return self._p3_12RF

	@property
	def m1(self) -> np.ndarray:
		'''Mass of particle 1'''
		if self._m1 is None:
			self._m1 = math.sqrt(lorentz.lp(self.p1, self.p1))
		return self._m1

	@property
	def m2(self) -> np.ndarray:
		'''Mass of particle 2'''
		if self._m2 is None:
			self._m2 = math.sqrt(lorentz.lp(self.p2, self.p2))
		return self._m2

	@property
	def m3(self) -> np.ndarray:
		'''Mass of particle 3'''
		if self._m3 is None:
			self._m3 = math.sqrt(lorentz.lp(self.p3, self.p3))
		return self._m3

	@property
	def p1(self) -> np.ndarray:
		'''4-momenta of the first particle of the shape (4, events)'''
		if self._p1 is None:
			log.raiseException(Exception, '`HadronicCurrents` object has no events set')
		return self._p1

	@property
	def p2(self) -> np.ndarray:
		'''4-momenta of the second particle of the shape (4, events)'''

		if self._p2 is None:
			log.raiseException(Exception, '`HadronicCurrents` object has no events set')
		return self._p2

	@property
	def p3(self) -> np.ndarray:
		'''4-momenta of the third particle of the shape (4, events)'''

		if self._p3 is None:
			log.raiseException(Exception, '`HadronicCurrents` object has no events set')
		return self._p3


	@property
	def p12(self) -> np.ndarray:
		'''4-momenta of particles 1 + 2 of the shape (4, events)'''

		if self._p12 is None:
			self._p12 = self.p1 + self.p2
		return self._p12

	@property
	def p13(self) -> np.ndarray:
		'''4-momenta of particles 1 + 3 of the shape (4, events)'''

		if self._p13 is None:
			self._p13 = self.p1 + self.p3
		return self._p13

	@property
	def p23(self) -> np.ndarray:
		'''4-momenta of particles 2 + 3 of the shape (4, events)'''

		if self._p23 is None:
			self._p23 = self.p2 + self.p3
		return self._p23

	@property
	def p123(self) -> np.ndarray:
		'''4-momenta of particles 1 + 2 + 3 of the shape (4, events)'''

		if self._p123 is None:
			self._p123 = self.p12 + self.p3
		return self._p123

	@property
	def s12(self) -> np.ndarray:
		'''Squared 4-momenta of (12)
		'''
		if self._s12 is None:
			self._s12 = lorentz.lp(self.p12, self.p12)
		return self._s12

	@property
	def s13(self) -> np.ndarray:
		'''Squared 4-momenta of (13)
		'''
		if self._s13 is None:
			self._s13 = lorentz.lp(self.p13, self.p13)
		return self._s13

	@property
	def s23(self) -> np.ndarray:
		'''Squared 4-momenta of (23)
		'''
		if self._s23 is None:
			self._s23 = lorentz.lp(self.p23, self.p23)
		return self._s23

	@property
	def s123(self) -> np.ndarray:
		'''Squared 4-momenta of (123)
		'''
		if self._s123 is None:
			self._s123 = lorentz.lp(self.p123, self.p123)
		return self._s123

	@property
	def m12(self) -> np.ndarray:
		'''Mass of subsystem (12)'''

		if self._m12 is None:
			self._m12 = math.sqrt(self.s12)
		return self._m12

	@property
	def m13(self) -> np.ndarray:
		'''Mass of subsystem (13)'''
		if self._m13 is None:
			self._m13 = math.sqrt(self.s13)
		return self._m13

	@property
	def m23(self) -> np.ndarray:
		'''Mass of subsystem (23)'''

		if self._m23 is None:
			self._m23 = math.sqrt(self.s23)
		return self._m23

	@property
	def m123(self) -> np.ndarray:
		'''Mass of subsystem (123)'''

		if self._m123 is None:
			self._m123 = math.sqrt(self.s123)
		return self._m123


	def __calcHelicityAngles_12(self)->None:
		if self._helicityAngles_12 is None:
			self._helicityAngles_12 = calculateIsobarTreeHelicityAngles([self.p1, self.p2, self.p3], momentaInCMS=self._momentaInCMS)

	@property
	def helicityRF__cosTheta_12__123(self) -> np.ndarray:
		'''Helicity angle :math:`\\cos(\\theta)` of spin analyzer (12) in rest frame (123)
		'''
		self.__calcHelicityAngles_12()
		return self._helicityAngles_12['cosTheta_12__123']

	@property
	def helicityRF__phi_12__123(self) -> np.ndarray:
		'''Helicity angle :math:`\\phi` of spin analyzer (12) in rest frame (123)
		'''
		self.__calcHelicityAngles_12()
		return self._helicityAngles_12['phi_12__123']

	@property
	def helicity__cosTheta_1__12(self) -> np.ndarray:
		'''Helicity angle :math:`\\cos(\\theta)` of spin analyzer (1) in rest frame (12)
		'''
		self.__calcHelicityAngles_12()
		return self._helicityAngles_12['cosTheta_1__12']

	@property
	def helicityRF__phi_1__12(self) -> np.ndarray:
		'''Helicity angle :math:`\\phi` of spin analyzer (1) in rest frame (12)
		'''
		self.__calcHelicityAngles_12()
		return self._helicityAngles_12['phi_1__12']


	def __calcHelicityAngles_13(self)->None:
		if self._helicityAngles_13 is None:
			self._helicityAngles_13 = calculateIsobarTreeHelicityAngles([self.p1, self.p3, self.p2], momentaInCMS=self._momentaInCMS)

	@property
	def helicityRF__cosTheta_13__123(self) -> np.ndarray:
		'''Helicity angle :math:`\\cos(\\theta)` of spin analyzer (13) in rest frame (123)
		'''
		self.__calcHelicityAngles_13()
		return self._helicityAngles_13['cosTheta_12__123']

	@property
	def helicityRF__phi_13__123(self) -> np.ndarray:
		'''Helicity angle :math:`\\phi` of spin analyzer (13) in rest frame (123)
		'''
		self.__calcHelicityAngles_13()
		return self._helicityAngles_13['phi_12__123']

	@property
	def helicity__cosTheta_1__13(self) -> np.ndarray:
		'''Helicity angle :math:`\\cos(\\theta)` of spin analyzer (1) in rest frame (13)
		'''
		self.__calcHelicityAngles_13()
		return self._helicityAngles_13['cosTheta_1__12']

	@property
	def helicityRF__phi_1__13(self) -> np.ndarray:
		'''Helicity angle :math:`\\phi` of spin analyzer (1) in rest frame (13)
		'''
		self.__calcHelicityAngles_13()
		return self._helicityAngles_13['phi_1__12']

def kallen(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
	'''
	General Källen function as shown in: https://en.wikipedia.org/wiki/K%C3%A4ll%C3%A9n_function
	Args:

		x,y,z:      Inputs to be used in calculation

	Returns:
		np.ndarray: Result of calculation
	'''

	return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*z*x

def cosTheta_12(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray, m12: np.ndarray, m13: np.ndarray, m123: np.ndarray) -> np.ndarray:
	'''
	Cosine of the scattering angle between particle 1 and 3 in the 12 isobar restframe: See Appendix A in DOI: 10.1103/PhysRevD.101.034033

    Value of Källen function should not become negative, but might due to numeric fluctuations. --> Use sqrtZeroBelowZero
	'''
	num = (2*m12**2*(m13**2-m3**2-m1**2)-(m12**2+m1**2-m2**2)*(m123**2-m12**2-m3**2))
	den = math.sqrtZeroBelowZero(kallen(m123**2,m3**2,m12**2)) * math.sqrtZeroBelowZero(kallen(m12**2,m1**2,m2**2))
	return num / den
