
import numpy as np

from .... import lorentz
from .... import math
from ...._constants import Constants as C

from .._kinematics import Kinematics
from ...utils import calculateIsobarTreeHelicityAngles


# pylint: disable=invalid-name,attribute-defined-outside-init,too-many-public-methods

class tau3Kinematics(Kinematics):
	"""
		Calculates the three-body kinematics of the :math:`\\tau` decay to three particles.
	"""

	def __init__(self, p1: np.ndarray = None, p2: np.ndarray = None ,p3: np.ndarray = None,
			  ECMS: np.ndarray = None, momentaInCMS: bool = False) -> None:
		"""_summary_

		Args:
			p1 (np.ndarray, optional): 4-momentum of particle 1 of the shape ((E, px, py, px), events).
			p2 (np.ndarray, optional): 4-momentum of particle 2 of the shape ((E, px, py, px), events).
			p3 (np.ndarray, optional): 4-momentum of particle 3 of the shape ((E, px, py, px), events).
			momentaInCMS (bool, optional):  True if the momenta are given in their CMS. In this case, the helicity angles of the
											decay of the full (1,2,3) system cannot be calculated and the z axis is used as
											reference plane of the subsystem decays.
		"""
		super().__init__(p1, p2, p3, momentaInCMS)
		self.setEvents(p1, p2, p3, ECMS)


	def setEvents(self, p1: np.array, p2: np.ndarray, p3: np.ndarray, ECMS: np.ndarray = None) -> None:
		'''Sets event data (momenta and CMS energy).

		:param p1: 4-momentum of particle 1
		:type p1: np.array
		:param p2: 4-momentum of particle 2
		:type p2: np.ndarray
		:param p3: 4-momentum of particle 3
		:type p3: np.ndarray
		:param ECMS: Center of mass energy, defaults to None
		:type ECMS: np.ndarray, optional
		'''
		super().setEvents(p1, p2, p3)
		self._ECMS = ECMS


	def reset(self):
		'''Resets Euler angles and helicity angles to `None`
		'''
		super().reset()
		self._kuhnEulerAngles = None
		self._helicityCMS__cosTheta_123_123nu = None
		self._helicity__cosTheta_123_123nu = None


	def _calculateKuhnEulerAngles(self):
		'''Calculate Euler angles :math:`\\cos(\\beta)` and :math:`\\gamma`.
		'''
		if self._kuhnEulerAngles is None:
			self._kuhnEulerAngles = {}
			self._kuhnEulerAngles['cosBeta'], self._kuhnEulerAngles['gamma'] = getKuhnEulerAngles(self.p1, self.p2, self.p3)

	@property
	def kuhnEulerAngleCosBeta(self) -> np.ndarray:
		'''The Euler angle :math:`\\cos(\\beta)`.
		'''
		self._calculateKuhnEulerAngles()
		return self._kuhnEulerAngles['cosBeta']

	@property
	def kuhnEulerAngleGamma(self) -> np.ndarray:
		'''The Euler angle :math:`\\gamma`.
		'''
		self._calculateKuhnEulerAngles()
		return self._kuhnEulerAngles['gamma']


	@property
	def helicity__cosTheta_123__123nu(self) -> np.ndarray:
		'''The helicity angle :math:`\\cos(\\theta_H)` of the :math:`\\tau` decay in the :math:`\\tau` rest frame
		'''
		if self._helicity__cosTheta_123_123nu is None:
			self._helicity__cosTheta_123_123nu = calcTauHelicityCosTheta_tauRF(self._ECMS/2., C.M.tau**2, self.p123[0], self.s123)
		return self._helicity__cosTheta_123_123nu

	@property
	def helicityCMS__cosTheta_123__123nu(self) -> np.ndarray:
		'''The helicity angle :math:`\\cos(\\theta_H)` of the :math:`\\tau` decay in the CMS rest frame
		'''

		if self._helicityCMS__cosTheta_123_123nu is None:
			self._helicityCMS__cosTheta_123_123nu = calcTauHelicityCosTheta_CMS(self._ECMS/2., C.M.tau**2, self.p123[0], self.s123)
		return self._helicityCMS__cosTheta_123_123nu



class KinematicsMCT(tau3Kinematics):
	"""
 		Generates the three-body kinematics of the tau decay to 3 particles using additional Monte-Carlo Truth information.
	"""

	def __init__(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
	             nu: np.ndarray, tau: np.ndarray) -> None:
		"""_summary_

		Args:
			p1 (np.ndarray): 4-momentum of particle 1 of the shape ((E, px, py, px), events).
			p2 (np.ndarray): 4-momentum of particle 2 of the shape ((E, px, py, px), events).
			p3 (np.ndarray): 4-momentum of particle 3 of the shape ((E, px, py, px), events).
			nu (np.ndarray): 4-momentum of neutrinoMCT of the shape ((E, px, py, px), events).
			tau (np.ndarray): 4-momentum of tauMCT of the shape ((E, px, py, px), events).
		"""
		super().__init__(p1, p2, p3)
		self.setEvents(p1, p2, p3, nu, tau)

	def setEvents(self, # pylint: disable=arguments-renamed
	              p1: np.ndarray,
	              p2: np.ndarray,
	              p3: np.ndarray,
	              nu: np.ndarray = None,
               	  tau: np.ndarray = None):
		'''Sets event data including :math:`\\nu` and :math:`\\tau` momenta.

		:param p1: 4-momentum of particle 1
		:type p1: np.array
		:param p2: 4-momentum of particle 2
		:type p2: np.ndarray
		:param p3: 4-momentum of particle 3
		:type p3: np.ndarray
		:param nu: 4-momentum of neutrinoMCT, defaults to None
		:type nu: np.ndarray, optional
		:param tau: 4-momentum of tauMCT, defaults to None
		:type tau: np.ndarray, optional
		'''
		super().setEvents(p1, p2, p3)
		self._nu = nu
		self._tau = tau
		if nu is not None:
			self._ECMS = self.p123nu[0]

	def reset(self):
		'''Resets class variables to `None`.
		'''
		super().reset()
		self._nu = None
		self._tau = None
		self._mTau = None
		self._p123nu = None
		self._m123nu = None
		self._helicityAngles_all_12 = None
		self._helicityAngles_all_13 = None
		self._helicityCMS__cosTheta_123_123nu = None
		self._helicity__cosTheta_123_123nu = None
		self._helicityCMS__alpha = None

	@property
	def pNu(self) -> np.ndarray:
		'''Returns 4 - momentum of the neutrino.
		'''
		return self._nu

	@property
	def pTau(self) -> np.ndarray:
		'''Returns 4 - momentum of the tauon.
		'''
		return self._tau

	@property
	def mTau(self) -> np.ndarray:
		'''Returns the invariant mass of the tauon, calculated from its  4 - momentum.
		'''
		if self._mTau is None:
			self._mTau = math.sqrt(lorentz.lp(self.pTau, self.pTau))
		return self._mTau

	@property
	def p123nu(self) -> np.ndarray:
		'''Returns total momentum of (123) plus the momentum of the neutrino
		'''
		if self._p123nu is None:
			self._p123nu = self.p123 + self.pNu
		return self._p123nu

	@property
	def m123nu(self) -> np.ndarray:
		'''Returns the invariant mass of the (123) system.
		'''
		if self._m123nu is None:
			self._m123nu = math.sqrt(lorentz.lp(self.p123nu, self.p123nu))
		return self._m123nu

	def _calculateKuhnEulerAngles(self):
		'''Calculates three Euler Angles, :math:`\\alpha, \\cos(\\beta)` and :math:`\\gamma`.
		'''
		if self._kuhnEulerAngles is None:
			self._kuhnEulerAngles = {}
			self._kuhnEulerAngles['alpha'], self._kuhnEulerAngles['cosBeta'], self._kuhnEulerAngles['gamma'] = getKuhnEulerAngles(self.p1, self.p2, self.p3, self.pTau) # pylint: disable=unbalanced-tuple-unpacking

	@property
	def kuhnEulerAngleAlpha(self):
		'''Returns Euler angle :math:`\\alpha`.
		'''
		self._calculateKuhnEulerAngles()
		return self._kuhnEulerAngles['alpha']

	def __calcHelicityAnglesAll_12(self) -> None:
		if self._helicityAngles_all_12 is None:
			self._helicityAngles_all_12 = calculateIsobarTreeHelicityAngles(
			    [self.p1, self.p2, self.p3, self.pNu])

	def __calcHelicityAnglesAll_13(self) -> None:
		if self._helicityAngles_all_13 is None:
			self._helicityAngles_all_13 = calculateIsobarTreeHelicityAngles(
			    [self.p1, self.p3, self.p2, self.pNu])

	@property
	def helicity__cosTheta_123__123nu(self) -> np.ndarray:
		'''The helicity angle :math:`\\cos(\\theta_H)` of the :math:`\\tau` decay in the :math:`\\tau` rest frame.
		'''
		if self._helicity__cosTheta_123_123nu is None:
			self._helicity__cosTheta_123_123nu = calcTauHelicityCosTheta_tauRF(self.pTau[0], self.mTau**2, self.p123[0], self.m123**2)
		return self._helicity__cosTheta_123_123nu

	@property
	def helicityCMS__cosTheta_123__123nu(self) -> np.ndarray:
		'''The helicity angle :math:`\\cos(\\theta_H)` of the :math:`\\tau` decay in the CMS rest frame.

		'''
		if self._helicityCMS__cosTheta_123_123nu is None:
			self._helicityCMS__cosTheta_123_123nu = calcTauHelicityCosTheta_CMS(self.pTau[0], self.mTau**2, self.p123[0], self.m123**2)
		return self._helicityCMS__cosTheta_123_123nu

	@property
	def helicityCMS__alpha(self) -> np.ndarray:
		"""Angle of the tau momentum in the plane perpendicular to the (123) momentum in the CMS frame
		"""
		if self._helicityCMS__alpha is None:
			_, pOrtho1, pOrtho2 = getTauMomentumComponentsCMS(self.p123, self.pTau[0], self.mTau)
			self._helicityCMS__alpha = math.arctan2(math.einsum('i...,i...->...', self.pTau[1:], pOrtho2[1:])/math.sqrt(math.sum(self.pTau[1:]**2, axis=0)
                                                                                                               *math.sum(pOrtho2[1:]**2, axis=0)),
			                                                 math.einsum('i...,i...->...', self.pTau[1:], pOrtho1[1:])
                                                    /math.sqrt(math.sum(self.pTau[1:]**2, axis=0)*math.sum(pOrtho1[1:]**2, axis=0)))
		return self._helicityCMS__alpha

	@property
	def helicity__phi_123__123nu(self) -> np.ndarray:
		'''The helicity angle :math:`\\phi` with spin analyzer (123) in the rest frame of (123 :math:`\\nu`)
		'''
		self.__calcHelicityAnglesAll_12()
		return self._helicityAngles_all_12['phi_123__1234']

	@property
	def helicity__cosTheta_12__123(self) -> np.ndarray:
		'''The helicity angle :math:`\\cos(\\theta)` with spin analyzer (12) in the rest frame of (123)
		'''
		self.__calcHelicityAnglesAll_12()
		return self._helicityAngles_all_12['cosTheta_12__123']

	@property
	def helicity__phi_12__123(self) -> np.ndarray:
		'''The helicity angle :math:`\\phi` with spin analyzer (12) in the rest frame of (123)
		'''
		self.__calcHelicityAnglesAll_12()
		return self._helicityAngles_all_12['phi_12__123']

	@property
	def helicity__cosTheta_13__123(self) -> np.ndarray:
		'''The helicity angle :math:`\\cos(\\theta)` with spin analyzer (13) in the rest frame of (123)
		'''
		self.__calcHelicityAnglesAll_13()
		return self._helicityAngles_all_13['cosTheta_12__123']

	@property
	def helicity__phi_13__123(self) -> np.ndarray:
		'''The helicity angle :math:`\\phi` with spin analyzer (13) in the rest frame of (123)
		'''
		self.__calcHelicityAnglesAll_13()
		return self._helicityAngles_all_13['phi_12__123']

	@property
	def helicity__phi_1__12(self) -> np.ndarray:
		'''The helicity angle :math:`\\phi` with spin analyzer (1) in the rest frame of (12)
		'''
		self.__calcHelicityAnglesAll_12()
		return self._helicityAngles_all_12['phi_1__12']

	@property
	def helicity__phi_1__13(self) -> np.ndarray:
		'''The helicity angle :math:`\\phi` with spin analyzer (1) in the rest frame of (13)
		'''
		self.__calcHelicityAnglesAll_13()
		return self._helicityAngles_all_13['phi_1__12']

def getKuhnEulerAngles(pCMS1, pCMS2, pCMS3, pCMSTau = None):
	"""Calculates the Euler angles :math:`\\cos(\\beta)`, :math:`\\gamma` and :math:`\\alpha` of the tau to three pion decay plane.
 	The angle is only calculated if the four momentum of the tau is provided.
  	The tau momentum can be known from the MCT information.
	Formulas taken from J.H. Kuhn and E. Mirkes, Z.Phys.C 56 (1992) 661-672
	DOI: 10.1007/BF01474741, URL: https://lib-extopc.kek.jp/preprints/PDF/1992/9207/9207509.pdf

	Args:
		pCMS1 (np.ndarray): Four momentum of oppositely charged pion in CMS-frame
		pCMS2 (np.ndarray): Four momentum of same charge pion in CMS-frame. Momentum of pCMS2 is smaller than of pCMS3
		pCMS3 (np.ndarray): Four momentum of same charge pion in CMS-frame.
		pCMSTau (np.ndarray, optional): Four momentum of tau (known from MCT) in CMS-frame.

	Returns:
		tuple
		- If 'pCMSTau' is None: np.ndarray, np.ndarray: Helicity angles :math:`\\cos(\\beta)`, :math:`\\gamma`
		- If 'pCMSTau' is provided: np.ndarray, np.ndarray, np.ndarray: Helicity angles :math:`\\alpha`, :math:`\\cos(\\beta)`, :math:`\\gamma`
	"""
	threePiMomentum = pCMS1 + pCMS2 + pCMS3
	boost = lorentz.getBoostToRestFrame(threePiMomentum)
	pOrderRF1 = lorentz.applyBoost(boost, pCMS1)
	pOrderRF2 = lorentz.applyBoost(boost, pCMS2)
	pOrderRF3 = lorentz.applyBoost(boost, pCMS3)

	# nCMS
	momentumCMS = pCMS1[1:,:] + pCMS2[1:,:] + pCMS3[1:,:]
	nCMS = -momentumCMS / math.sqrt(math.sum(momentumCMS**2, axis=0))

	# n_vertical
	crossProductQ2Q3 = math.cross(pOrderRF2[1:,:], pOrderRF3[1:,:], axis=0)
	nPerp = crossProductQ2Q3 / math.sqrt(math.sum(crossProductQ2Q3**2, axis=0))

	# cos beta
	angleCosBeta = math.sum(nCMS * nPerp,axis=0)

	# Unit vector of pOrderRF1
	nOrderRF1 = pOrderRF1[1:,:]/math.sqrt(math.sum(pOrderRF1[1:,:]**2, axis=0))

	# sin gamma
	angleSinGamma = math.sum(math.cross(nCMS, nPerp, axis=0) * nOrderRF1, axis=0) / math.sqrt(math.sum(math.cross(nCMS, nPerp, axis=0)**2, axis=0))

	# cos gamma
	angleCosGamma = - math.sum(nCMS * nOrderRF1, axis=0) / math.sqrt(math.sum(math.cross(nCMS, nPerp, axis=0)**2, axis=0))

	# gamma
	angleGamma = math.arctan2(angleSinGamma, angleCosGamma)

	if pCMSTau is None:
		return angleCosBeta, angleGamma

	pRFTau = lorentz.applyBoost(boost, pCMSTau)[1:,:]
	nRFTau = pRFTau/math.sqrt(math.sum(pRFTau**2, axis=0))
	divisor = math.sqrt(math.sum(math.cross(nCMS, nRFTau, axis=0)**2, axis=0))*math.sqrt(math.sum(math.cross(nCMS, nPerp, axis=0)**2, axis=0))
	angleCosAlpha = math.sum((math.cross(nCMS, nRFTau, axis=0) * math.cross(nCMS, nPerp, axis=0)), axis=0) / divisor
	angleSinAlpha = -math.sum((nRFTau * math.cross(nCMS, nPerp, axis=0)), axis=0) / divisor
	angleAlpha = math.arctan2(angleSinAlpha, angleCosAlpha)
	return angleAlpha, angleCosBeta, angleGamma



def calcTauHelicityCosTheta_CMS(tauE: np.ndarray, tauM2: np.ndarray, hadronE: np.ndarray, hadronM2: np.ndarray) -> np.ndarray:
	'''
	Calculates the helicity angle :math:`\\cos(\\theta_H)`, where :math:`\\theta_H` is the helicity angle of the tau decay in the :math:`e^+e^-` CMS

	:param tauE: Energy of the tauon in CMS
	:param tauM2: squared mass of the tauon
	:param hadronE: Energy of the hadronic system in CMS
	:param hadronM2: squared mass of the hadronic system
	'''
	tauAbsP = math.sqrt(tauE**2 - tauM2)
	hadAbsP = math.sqrt(hadronE**2 - hadronM2)
	scalarProduct = (tauM2 + hadronM2)/2.
	cosT = (hadronE*tauE - scalarProduct)/(tauAbsP * hadAbsP)
	return cosT



def calcTauHelicityCosTheta_tauRF(tauE: np.ndarray, tauM2: np.ndarray, hadronE: np.ndarray, hadronM2: np.ndarray) -> np.ndarray:
	'''
	Calculates the :math:`\\cos(\\theta_H)`, where :math:`\\theta_H` is the helicity angle of the :math:`\\tau` decay in the :math:`\\tau` rest frame

	The z-axis is defined by the direction of the tau in the :math:`e^+e^-` CMS

	:param tauE: Energy of the tauon in CMS
	:param tauM2: squared mass of the :math:`\\tau`
	:param hadronE: Energy of the hadronic system in CMS
	:param hadronM2: squared mass of the hadronic system
	'''
	energyRatioX = 2*hadronE/(2*tauE)
	cosT = (2*energyRatioX*tauM2-tauM2-hadronM2)/((tauM2-hadronM2)*math.sqrt(1-tauM2/tauE**2))
	return cosT



def getTauMomentumComponentsCMS(pHadron, tauE, tauM, orthoDirection = np.array([0.,0.,1.])):
	"""
	Reconstructs the tau momenta up to a single free angle :math:`\\alpha` from the hadron momentum, the known tau energy and mass, such that:

	.. math::
	    p_{\\tau} = p_{\\tau}^{\\parallel} + \\cos(\\alpha) \\cdot p_\\tau^{\\perp1} + \\sin(\\alpha) \\cdot p_\\tau^{\\perp2}

	All given in the :math:`e^+ e^-` CM frame.

	:param pHadron: Hadronic four-momentum [E,px,py,pz]
	:param tauE: Energy of the tauon
	:param tauM: Mass of the tauon
	:param orthoDirection: direction used to define the orthogonal coordinate system
	"""
	tauAbsP = math.sqrt(tauE**2 - tauM**2)
	hadE = pHadron[0]
	hadAbsP = math.sqrt(math.sum(pHadron[1:4]**2, axis = 0))
	hadM2   = hadE**2 - hadAbsP**2

	# helicity angle in CMS
	cosT = calcTauHelicityCosTheta_CMS(tauE, tauM**2, hadE, hadM2)
	sinT = math.sqrt(1. - cosT**2)

	pTauParallelUnit   = math.empty(pHadron.shape, pHadron)
	pTauParallelUnit[0]   = 0.
	pTauParallelUnit[1:]  = pHadron[1:]/hadAbsP[None,:]

	pTauParallel   = math.empty(pHadron.shape, pHadron)
	pTauParallel[0]   = tauE
	pTauParallel[1:]  = pTauParallelUnit[1:] * ( cosT*tauAbsP )[None,:]

	pTauOrtho1Unit   = math.empty(pHadron.shape, pHadron)
	pTauOrtho1Unit[0]   = 0.
	pTauOrtho1Unit[1:]  = np.cross(pTauParallelUnit[1:],orthoDirection, axis=0)
	pTauOrtho1Unit[1:] /= math.sqrt(np.sum(pTauOrtho1Unit[1:]**2, axis = 0))[None,:]

	pTauOrtho2Unit   = math.empty(pHadron.shape, pHadron)
	pTauOrtho2Unit[0]   = 0.
	# No normalization needed here, since pTauParallelUnit and pTauOrtho1Unit are orthogonal and normalized
	pTauOrtho2Unit[1:]  = np.cross(pTauParallelUnit[1:],pTauOrtho1Unit[1:], axis=0)

	return pTauParallel, pTauOrtho1Unit*(sinT*tauAbsP)[None,:] , pTauOrtho2Unit*(sinT*tauAbsP)[None,:]
