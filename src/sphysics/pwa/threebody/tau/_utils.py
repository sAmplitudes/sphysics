# coding: utf-8
'''
Created on Wednesday 23 03 2022
@author: Stefan Wallner
@description: Utility functions for tau -> 3body PWA
'''

from __future__ import absolute_import, print_function, division

import numpy as np

from .... import math
from .... import generators
from .... import eventselection
from .... import lorentz
from ...._constants import Constants as C
from ....utils import Logger


log = Logger("tau")


class EventsTau2ThreeChargedPi(eventselection.VariablesBase):
	'''
	Events of the reaction :math:`\\tau^{\\mp} \\to \\pi^{\\pm} \\pi^{\\mp} \\pi^{\\mp} \\nu_\\tau`

	pi1MP, pi2PM, pi3PM are ordered according to the reaction given above,
	i.e. the first two pions have opposite charge and form the isobar system and the second and the last pions are exchanged for the bose symmetrization
	'''
	def __init__(self, tauMP_E:float=None, tauMP_M:float=None) -> None:
		super().__init__()
		self.pi1PM = np.empty((4,0), dtype=np.float64)
		self.pi2MP = np.empty((4,0), dtype=np.float64)
		self.pi3MP = np.empty((4,0), dtype=np.float64)
		self.tauMP = np.empty((4,0), dtype=np.float64)
		self.nu = np.empty((4,0), dtype=np.float64)
		self.isParticle = np.empty(0, dtype=bool)
		self.tauMP_E = tauMP_E # energy of the tau
		self.tauMP_M = tauMP_M # mass of the tau

	# def appendTauPlusEvents(self, piM, piP1, piP2, tauP = None, nu = None):
	# 	''''
	# 	Add event of the anti-particle reactin, i.e. tau^+ decau.
	# 	'''
	# 	if tauP is not None:
	# 		self.tauPM = np.concatenate([self.tauPM,  tauP], axis=-1)
	# 	if nu is not None:
	# 		self.nu = np.concatenate([self.nu,  nu], axis=-1)
	# 	self.pi1PM = np.concatenate([self.pi1PM,  piM], axis=-1)
	# 	self.pi2MP = np.concatenate([self.pi2MP,  piP1], axis=-1)
	# 	self.pi3MP = np.concatenate([self.pi3MP,  piP2], axis=-1)
	# 	self.isParticle = np.concatenate([self.isParticle,  np.zeros(piM.shape[-1], dtype=self.isParticle.dtype)], axis=-1)

	# def appendTauMinusEvents(self, piP, piM1, piM2, tauM = None, nu = None):
	# 	''''
	# 	Add event of the particle reaction, i.e. tau^- decau.
	# 	'''
	# 	if tauM is not None:
	# 		self.tauPM = np.concatenate([self.tauPM,  tauM], axis=-1)
	# 	if nu is not None:
	# 		self.nu = np.concatenate([self.nu,  nu], axis=-1)
	# 	self.pi1PM = np.concatenate([self.pi1PM,  piP], axis=-1)
	# 	self.pi2MP = np.concatenate([self.pi2MP,  piM1], axis=-1)
	# 	self.pi3MP = np.concatenate([self.pi3MP,  piM2], axis=-1)
	# 	self.isParticle = np.concatenate([self.isParticle,  np.ones(piP.shape[-1], dtype=self.isParticle.dtype)], axis=-1)


	# def variablesReco2Events(variables, tauPM_E, tauPM_M, addVariables=None) -> EventsTau2ThreeChargedPi:
	# """Generate an Events object form the given `variables` object.

	# The `variables` object must have certain branches.
	# The ordering of events in the return Events object is the same is in the `variables` object.
	# The reconstructed quantities are used!

	# :param variables: Input variables object from which the tau events are loaded
	# """
	# raise NotImplementedError("Needs to be re-implemented")
	# events = _variables2Events(variables.getTotalNevents(),
	#                            variables.track1_3prong_charge, variables.track2_3prong_charge, variables.track3_3prong_charge,
	#                            variables.track1_3prong_p_CMS, variables.track2_3prong_p_CMS, variables.track3_3prong_p_CMS,
	#          tauPM_E, tauPM_M
	#        )
	# if addVariables:
	# 	for element in addVariables:
	# 		if not isinstance(element, tuple):
	# 			element = (element, element)
	# 		variable, dstName = element
	# 		events.addVariable(dstName)
	# 		events[dstName] = variables[variable]
	# events.check()
	# return events


def variablesMCT2Events(sample: eventselection.Variables, prefix: str = 'genCMS_') -> EventsTau2ThreeChargedPi:
	"""Generate an Events object form the given `sample` object with the prefix `genCMS_`.
	"""
	m = math.sqrt(lorentz.lp(sample[f'{prefix}tauMP'], sample[f'{prefix}tauMP']))
	events = EventsTau2ThreeChargedPi(tauMP_E=sample[f'{prefix}tauMP'][0], tauMP_M=m)
	events.pi1PM = sample[f'{prefix}pi1PM']
	events.pi2MP = sample[f'{prefix}pi2MP']
	events.pi3MP = sample[f'{prefix}pi3MP']
	events.tauMP = sample[f'{prefix}tauMP']
	events.nu    = sample[f'{prefix}nu']
	events.isParticle = sample[f'{prefix}isParticle']
	events.check()
	return events

def variablesReco2Events(sample: eventselection.Variables, prefix: str = '') -> EventsTau2ThreeChargedPi:
	"""Generate an Events object form the given `sample` object.
	"""
	m = C.M.tau
	events = EventsTau2ThreeChargedPi(tauMP_E=sample['Ecms']/2, tauMP_M=m)
	events.pi1PM = sample[f'{prefix}pi1PM']
	events.pi2MP = sample[f'{prefix}pi2MP']
	events.pi3MP = sample[f'{prefix}pi3MP']
	events.isParticle = sample[f'{prefix}isParticle']

	print(events.nEvents)
	print(events.pi1PM)
	events.check()
	return events

def _variables2Events(nEvents,
                      track1_charge, track2_charge, track3_charge,
       track1_p_CMS, track2_p_CMS, track3_p_CMS,
       tauPM_E, tauPM_M) -> EventsTau2ThreeChargedPi:
	"""Generate an Events object form the given momenta and charges

	The ordering of events in the return Events object is the same is in the input variables.

	"""
	events = EventsTau2ThreeChargedPi(tauMP_E=tauPM_E, tauMP_M=tauPM_M)
	events.pi1PM = np.full((4,nEvents), np.nan)
	events.pi2MP = np.full((4,nEvents), np.nan)
	events.pi3MP = np.full((4,nEvents), np.nan)
	events.isParticle = np.empty(nEvents, dtype=events.isParticle.dtype)


	### ensure the same ordering in events and variables!!

	#events.tauPM
	####
	# tau-
	# +--
	iSelect = (track1_charge==+1)&(track2_charge==-1)&(track3_charge==-1)
	events.pi1PM[...,iSelect] = track1_p_CMS[...,iSelect]
	events.pi2MP[...,iSelect] = track2_p_CMS[...,iSelect]
	events.pi3MP[...,iSelect] = track3_p_CMS[...,iSelect]
	events.isParticle[iSelect] = True

	# -+-
	iSelect = (track1_charge==-1)&(track2_charge==+1)&(track3_charge==-1)
	events.pi1PM[...,iSelect] = track2_p_CMS[...,iSelect]
	events.pi2MP[...,iSelect] = track1_p_CMS[...,iSelect]
	events.pi3MP[...,iSelect] = track3_p_CMS[...,iSelect]
	events.isParticle[iSelect] = True

	# --+
	iSelect = (track1_charge==-1)&(track2_charge==-1)&(track3_charge==+1)
	events.pi1PM[...,iSelect] = track3_p_CMS[...,iSelect]
	events.pi2MP[...,iSelect] = track1_p_CMS[...,iSelect]
	events.pi3MP[...,iSelect] = track2_p_CMS[...,iSelect]
	events.isParticle[iSelect] = True

	####
	# tau+
	# -++
	iSelect = (track1_charge==-1)&(track2_charge==+1)&(track3_charge==+1)
	events.pi1PM[...,iSelect] = track1_p_CMS[...,iSelect]
	events.pi2MP[...,iSelect] = track2_p_CMS[...,iSelect]
	events.pi3MP[...,iSelect] = track3_p_CMS[...,iSelect]
	events.isParticle[iSelect] = False

	# +-+
	iSelect = (track1_charge==+1)&(track2_charge==-1)&(track3_charge==+1)
	events.pi1PM[...,iSelect] = track2_p_CMS[...,iSelect]
	events.pi2MP[...,iSelect] = track1_p_CMS[...,iSelect]
	events.pi3MP[...,iSelect] = track3_p_CMS[...,iSelect]
	events.isParticle[iSelect] = False


	# ++-
	iSelect = (track1_charge==+1)&(track2_charge==+1)&(track3_charge==-1)
	events.pi1PM[...,iSelect] = track3_p_CMS[...,iSelect]
	events.pi2MP[...,iSelect] = track1_p_CMS[...,iSelect]
	events.pi3MP[...,iSelect] = track2_p_CMS[...,iSelect]
	events.isParticle[iSelect] = False

	if events.nEvents != nEvents:
		log.raiseException(Exception, "Number of events does not agree!")
	if np.any(np.isnan(events.pi1PM)) or np.any(np.isnan(events.pi2MP)) or np.any(np.isnan(events.pi3MP)) :
		log.raiseException(Exception, "Number of events does not agree!")

	return events




# def calcHelicityAnglesUsingCMSRefFrame(events, kinematics):
# 	'''
# 	Calculate the 2 two-body decay angles using the e^+e^- CMS as reference to construct the coordinate axes in the X reference frame, i.e.
# 	the z-axis is the direction of the x in the e^+e^- CMS and the reaction plane is given by the lab-frame z-axis (=CMS z-axis) and the z-axis of the X reference frame.

# 	The calculated angles are added to the kinematics.

# 	helicityAnglesCMS_cosThetaXSym: cosTheta of the 12 system in the X reference frame (symmetrized)
# 	helicityAnglesCMS_phiXSym: phi angle of the 12 system in the X reference frame (symmetrized)
# 	helicityAnglesCMS_cosTheta12Sym: cosTheta of pi1MP in the 12 reference frame (symmetrized)
# 	helicityAnglesCMS_phi12Sym: phi angle of pi1MP in the 12 reference frame (symmetrized)
# 	'''

# 	#################
# 	# Boost to 123 rest frame to calculate angles

# 	p12Sym = np.concatenate([kinematics.p12, kinematics.p13], axis=-1)
# 	xPMSym = np.concatenate([kinematics.xPM, kinematics.xPM], axis=-1)

# 	(cosThetaXSym, phiXSym), (boostX, p12Sym_xRF, zXRF) = calcHelicityAngles(xPMSym, p12Sym, np.array([0.,0.,0.,1.]))

# 	#################
# 	# Boost to 12Sym rest frame to calculate angles

# 	pi1Sym = np.concatenate([events.pi1MP, events.pi1MP], axis=-1)
# 	pi1Sym_xRF = math.einsum('ij...,j...->i...', boostX, pi1Sym)

# 	(cosTheta12Sym, phi12Sym), _ = calcHelicityAngles(p12Sym_xRF, pi1Sym_xRF, zXRF)

# 	kinematics.helicityAnglesCMS_cosThetaXSym = cosThetaXSym
# 	kinematics.helicityAnglesCMS_phiXSym = phiXSym
# 	kinematics.helicityAnglesCMS_cosTheta12Sym = cosTheta12Sym
# 	kinematics.helicityAnglesCMS_phi12Sym = phi12Sym



def outerProduct(vec):
	"""
	returns the outer products of an array of vectors
	@param vec input vectors [shape = (dim, nVectors) ==> output shape = (dim, dim, nVectors)]
	"""
	return np.einsum("ie,je->ije", vec, math.conjugate(vec))


def decomposeHermitianMatrix(mat: np.ndarray) -> np.ndarray:
	"""
	Decomposes a hermitian matrix into vectors, such, that
	vecs = decompose(mat) => mat = np.einsum("ime,ine->mne", vecs, conjugate(vecs))
	@param mat np.array of hermitian matrix of shape ((dim, dim, nEvents))
	"""
	dim = mat.shape[0]
	retVal = np.zeros_like(mat)
	retVal[0,:,:]  = mat[:,0,:]
	isZero = np.abs(mat[0]) < 1e-7
	for i in range(1,mat.shape[0]):
		if np.any(isZero[0] & ~isZero[i]):
			print(mat[0,i,isZero[0] & ~isZero[i]])
			raise Exception("The diagonal element is zero, but not the whole row is zero!")
	isNonZero = ~isZero[0]
	if np.any(mat[0,0,isNonZero] < 0.):
		print(mat[0,0,isNonZero][mat[0,0,isNonZero]<0])
		raise Exception("Diagonal elements must be larger than zero!")

	if dim == 1:
		return math.sqrt(mat)

	isLargerZero = mat[0,0] > 0.0
	retVal[0][:,isLargerZero] /= math.sqrt(mat[0,0,isLargerZero])[None,:]
	retVal[0][:,~isLargerZero]  = 0.0

	retVal[1:,1:,:] = decomposeHermitianMatrix(mat[1:,1:,:] - outerProduct(retVal[0,1:]))

	return retVal


def decomposeHermitianMatrix2(mat: np.ndarray) -> np.ndarray:
	"""
	Decomposes a hermitian matrix into vectors, such, that
	vecs = decompose(mat) => mat = np.einsum("ime,ine->mne", vecs, conjugate(vecs))
	@param mat np.array of hermitian matrix of shape ((dim, dim, nEvents))
	"""
	eigenValue, eigenVector = np.linalg.eigh(np.transpose(mat, (2,0,1)))

	if np.max(np.abs(eigenValue)) < -1e-10:
		raise Exception("Eigenvalues must be positive")

	eigenValue[eigenValue<0] = 0. # set eigenvalues that are negative, but numerically close to zero (see checks above), to zero

	eigenVector *= math.sqrt(eigenValue[:,None,:])

	return math.transpose(eigenVector, (2,1,0))


def calcDecayAmplitudesIntegratedNormIntegrals(model, xPM_M_range, nMCevents, energyCMS=None):
	'''
	Calculate the normalization integrals for the integrated decay amplitudes of the given model
	for particles and antiparticles.
	Returns the normalization integrals for the particle and antiparticle separately.

	:param nMCevents: Number of MC events that will be generated in order to calculate the integral
	:param ECM: Energy of the center-of-momentum system :math:`e^+e^-`. If `None` is given energy is set to ECM=M_upsilon4S
	'''
	if energyCMS is None:
		energyCMS=C.M.upsilon4S
	intParticle = np.zeros((model.nWaves, 1, 1), dtype=float)
	intAntiparticle = np.zeros((model.nWaves, 1, 1), dtype=float)
	batch_size = 100_000
	if nMCevents%batch_size != 0:
		log.raiseException(ValueError, "nMCevents must be multiple of 100,000")

	nBatches = max(nMCevents // batch_size, 1)
	for _ in range(nBatches):
		pTau, pnu, p1, p2, p3 = generators.tauToThreePi(batch_size, energyCMS**2, xPM_M_range)
		mcData = EventsTau2ThreeChargedPi(energyCMS/2., C.M.tau)
		mcData.pi1PM = p1
		mcData.pi2MP = p2
		mcData.pi3MP = p3
		mcData.tauMP = pTau
		mcData.nu    = pnu
		mcData.isParticle    = np.empty(batch_size, dtype=bool)

		mcData.isParticle[:] = True
		decayAmplParticle = model.calcDecayAmplitudesIntegrated(mcData, normalized=False)
		intParticle += math.einsum('aie->a', math.abs2(decayAmplParticle))[:,None,None]/batch_size

		mcData.isParticle[:] = False
		decayAmplAntiparticle = model.calcDecayAmplitudesIntegrated(mcData, normalized=False)
		intAntiparticle += math.einsum('aie->a', math.abs2(decayAmplAntiparticle))[:,None,None]/batch_size

	return intParticle/nBatches, intAntiparticle/nBatches


def setDecayAmplitudesIntegratedNormIntegrals(model, xPM_M_range, nMCevents, energyCMS=None):
	'''
	Calculate and set the normalization integrals for the integrated decay amplitudes of the given model.

	:param nMCevents: Number of MC events that will be generated in order to calculate the integral
	:param ECM: energy of the center-of-momentum system  :math:`e^+e^-`. If `None` is given, energy is set to ECM=M_upsilon4S
	'''
	model.setDecayAmplitudesIntegratedNormIntegrals(
	 *calcDecayAmplitudesIntegratedNormIntegrals(model, xPM_M_range, nMCevents, energyCMS=energyCMS),
	)


def calcIntegralMatrix(reconMC_decayAmplitudes: np.ndarray, genMC_nEvents: int, genMC_eventWeights: np.ndarray =None, reconMC_eventWeights: np.ndarray =None):
	'''
	Calculates the integral matrix from the given set of decay amplitudes

	:param reconMC_decayAmplitudes: Decay amplitudes of the reconMC dataset, of the form (nWaves, nIncoherentIndex, nEvents)
	:param genMC_nEvents: Number of events in genMC dataset (different from the number of events in reconMC dataset)
	:param genMC_eventWeights: Array with weights of genMC dataset that are used in the integral calculation
	:param reconMC_eventWeights: Array with weights of reconMC dataset that are used in the integral calculation
	'''
	if (genMC_eventWeights is None) and (reconMC_eventWeights is None):
		return math.einsum('aie,bie->ab', reconMC_decayAmplitudes, math.conjugate(reconMC_decayAmplitudes))/genMC_nEvents


	norm = genMC_eventWeights.size/genMC_nEvents/np.sum(genMC_eventWeights) # we do not assume a special normalization of the weights
	return math.einsum('aie,bie,e->ab', reconMC_decayAmplitudes, math.conjugate(reconMC_decayAmplitudes), reconMC_eventWeights)*norm
