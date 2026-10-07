# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Fitresult class, Created on Tuesday 29 03 2022
'''

from __future__ import absolute_import, print_function, division, annotations

import pickle
import numpy as np

from .._model import PWAModel
from ..._core.utilities import Environment #pylint: disable=import-error
from ..._core.pwa import FitResult, ResultCollection, equal #pylint: disable=import-error,unused-import
from ...utils import Logger

log = Logger("PWAFit")

# FitResult = core_pwa.FitResult

fitresult_origInit = FitResult.__init__
def createFitresult(couplings: np.ndarray, covarianceMatrix: np.ndarray, auxiliaryParameters:  np.ndarray, auxiliaryParameterNames: np.ndarray,
				negLogLikelihood: float, fitConverged: bool, covMatrixValide: bool, covMatrixMadePosDef: bool, resultObj) -> None:
	'''
	:param couplings: Parameter set of the fit
	:param negLogLikelihood: Optimized loss value
	:param auxiliaryParameters: Additional parameters (e.g. background yields)
	:param auxiliaryParameterNames: Additional parameter names
	:param covMatrix: Covariance matrix
	:param covMatrixIndices: Dictionary of parameter: index of parameter indices
	:param fitConverged: True if the fit attempt converged
	:param covMatrixValide: Covariance matrix of the fit is at a valid minimum
	:param covMatrixMadePosDef: Covariance matrix is not valid bad was forced to be positive definite
	:param resultObj: Result object specific to the fitter
	'''
	# serialize results object if it was not already serialized
	resultObjDump = pickle.dumps(resultObj) if not isinstance(resultObj, bytes) else resultObj
	return FitResult(
		                couplings, covarianceMatrix,
						negLogLikelihood, fitConverged, covMatrixValide, covMatrixMadePosDef,
						resultObjDump, auxiliaryParameters, auxiliaryParameterNames)

def fitresult_getResultObj(self):
	return pickle.loads(self.getResultObjDump())
FitResult.resultObj = property(fitresult_getResultObj, None, None, "Get result object of mimimizer.")



# ResultCollection = core_pwa.ResultCollection

resultcollection_origInit = ResultCollection.__init__
def resultcollection_init(self, model: PWAModel,
                                indicesCouplings: dict, indicesCouplingsCovariance: list,
								integralMatrixGen: np.ndarray, integralMatrixRecon: np.ndarray,*args):
	'''
	@param model: Fit model
	@param indicesCouplings: Mappling of wave name to index in list of couplings,
							each entry is a tuple with indices one or more
							complex-valued coupling amplitudes
	@param indicesCouplingsCovariance: Mappling of coupling index to index in matrix of
									covariance-matrix elements, each entry is a
									tuples (iRe, iIm) of indices to the
									covariance-matrix elements
	'''
	if isinstance(model, PWAModel):
		modelObj = pickle.dumps(model)
		resultcollection_origInit( self,
								model.name,
								model.description,
								modelObj,
								integralMatrixGen,
								integralMatrixRecon,
								indicesCouplings,
								indicesCouplingsCovariance,
								[],
								"",
								Environment())
	else: # full init (for unpickling)
		resultcollection_origInit(self, model, integralMatrixGen, integralMatrixRecon ,indicesCouplings, indicesCouplingsCovariance, *args)
ResultCollection.__init__ = resultcollection_init

def resultcollection_getModelObj(self):
	return pickle.loads(self.getModelObjDump())
ResultCollection.model = property(resultcollection_getModelObj, None, None, "The fit model")

def resultcollection_iterator(self):
	return iter(self.fitResults)
ResultCollection.__iter__ = resultcollection_iterator

# class FitResult:
# 	def __init__(self, couplings: np.ndarray, negLogLikelihood: float, covMatrix: np.ndarray,
# 	             fitConverged: bool, covMatrixValide: bool, covMatrixMadePosDef: bool, resultObj) -> None:
# 		'''
# 		@param couplings: Parameter set of the fit
# 		@param negLogLikelihood: Optimized loss value
# 		@param covMatrix: Covariance matrix
# 		@param covMatrixIndices: Dictionary of parameter: index of parameter indices
# 		@param fitConverged: True if the fit attempt converged
# 		@param covMatrixValide: Covariance matrix of the fit is at a valid minimum
# 		@param covMatrixMadePosDef: Covariance matrix is not valid bad was forced to be positive definite
# 		@param resultObj: Result object specific to the fitter
# 		'''
# 		self._negLogLikelihood = negLogLikelihood
# 		self._fitConverged = fitConverged
# 		self._covMatrixValide = covMatrixValide
# 		self._covMatrixMadePosDev = covMatrixMadePosDef
# 		self._resultObj = resultObj

# 		self._couplings = couplings

# 		self._covMatrix = covMatrix

# 	@property
# 	def negLogLikelihood(self):
# 		return self._negLogLikelihood

# 	@property
# 	def couplings(self):
# 		return self._couplings

# 	@property
# 	def covMatrix(self):
# 		return self._covMatrix


# 	@property
# 	def fitConverged(self):
# 		return self._fitConverged

# 	@property
# 	def covarianceMatrixValide(self):
# 		return self._covMatrixValide

# 	@property
# 	def covarianceMatrixMadePosDef(self):
# 		return self._covMatrixMadePosDev

# 	def isParameterAtLimits(self, fullParameterName):
# 		'''
# 		@return: 0 if not at limits, -1 if at lower limits, else 1. None if unknown.
# 		'''
# 		raise NotImplementedError()

# 	@property
# 	def resultObj(self):
# 		return self._resultObj

# 	def __lt__(self, other: FitResult) -> bool:
# 		if np.isnan(self.negLogLikelihood):
# 			return False
# 		if np.isnan(other.negLogLikelihood):
# 			return True
# 		return self.negLogLikelihood < other.negLogLikelihood


# 	def __eq__(self, other: FitResult) -> bool:
# 		'''
# 		Compare if two fit results are the same
# 		'''

# 		deltaMaxNegLogLikelihood = 1e-5

# 		same=True

# 		same &= math.abs(self.negLogLikelihood-other.negLogLikelihood) < deltaMaxNegLogLikelihood

# 		same &= math.max( math.real(self.couplings - other.couplings)/math.abs(self.couplings) ) < 1e-2
# 		same &= math.max( math.imag(self.couplings - other.couplings)/math.abs(self.couplings) ) < 1e-2

# 		return same



# class ResultCollection(list):
# 	'''
# 	Stores result of optimization
# 	'''
# 	def __init__(self, model: PWAModel,
# 	             indicesCouplings: dict, indicesCouplingsCovariance: dict,
# 	    *args, **kwargs):
# 		'''
# 		@param model: Fit model
# 		@param indicesCouplings: Mappling of wave name to index in list of couplings,
# 		                        each entry is a tuple with indices one or more
# 								complex-valued coupling amplitudes
# 		@param indicesCouplingsCovariance: Mappling of wave name to index in matrix of
# 		                                covariance-matrix elements, each entry is a tuple
# 										with tuples (iRe, iIm) of indices to the
# 										covariance-matrix elements
# 		'''
# 		list.__init__(self, *args, **kwargs)
# 		self._label = ""
# 		self._modelName = model.name
# 		self._modelDescription = model.description
# 		self._environment = Environment()
# 		self._indicesCouplings = indicesCouplings
# 		self._indicesCouplingsCovariance = indicesCouplingsCovariance

# 	@property
# 	def ordered(self):
# 		'''
# 		@return: Results orders such that the best result is at the beginning
# 		'''
# 		return sorted(self)

# 	@property
# 	def best(self):
# 		convertedOrdered = [ r for r in self.ordered if r.fitConverged ]
# 		return convertedOrdered[0] if convertedOrdered else None

# 	@property
# 	def smalestLossResult(self):
# 		return self.ordered[0] if self else None

# 	@property
# 	def label(self):
# 		return self._label

# 	@property
# 	def modelDescription(self):
# 		return self._modelDescription

# 	@property
# 	def modelName(self):
# 		return self._modelName


# 	def buildLabel(self):
# 		if self._label:
# 			log.raiseException(Exception, "Label of fit result already set to '{0}'!", self._label)
# 		label = hashlib.sha1()
# 		label.update(self._modelName)
# 		label.update(self._modelDescription)
# 		label.update(str(len(self)))
# 		self._label = label.hexdigest()[0:7]
# 		raise NotImplementedError("Not fully implemented")

# 	def getNBestResults(self) -> int:
# 		'''
# 		@return: Number of times the best result was found
# 		'''
# 		return np.sum([self.best==r for r in self])

# 	def check(self) -> bool:
# 		fractionOfAttemptsFoundBest = 0.1
# 		log.info("Checks and __eq__ of results not fully implemented!")

# 		checksOK = True

# 		for idx in self._indicesCouplings.values():
# 			if len(idx) != len(list(self._indicesCouplings.values())[0]):
# 				log.raiseException(Exception, "Number of coupling amplitude indices is not the same for all waves!")

# 		for idx in self._indicesCouplingsCovariance.values():
# 			if len(idx) != len(list(self._indicesCouplings.values())[0]):
# 				log.raiseException(Exception, "Number of covariance coupling amplitude indices different from the number of coupling amplitude indices for at least one wave!")
# 			for idxSector in idx:
# 				if len(idxSector) != 2:
# 					log.raiseException(Exception, "Number of covariance coupling amplitude indices per wave and sector must be two, i.e. iReal and iImag!")

# 		if self.getNBestResults() == 1:
# 			log.warning("Found best solution only once!")
# 			checksOK = False
# 		elif self.getNBestResults() <= len(self)*fractionOfAttemptsFoundBest:
# 			log.warning("Found best solution in less than {0}% of attempts!", fractionOfAttemptsFoundBest*100)
# 			checksOK = False

# 		if checksOK:
# 			log.success("Fit result is OK :)")
# 		return checksOK


# 	def getIntensity(self, waveName: str, result: FitResult=None) -> float:
# 		'''
# 		Calculate the intensity of the given partial
# 		If `result` is None, use the best result
# 		'''
# 		return math.real(self.getCouplingSpinDensityMatrix(waveName, waveName, result))


# 	def getIntensityUncertainty(self, waveName: str, result: FitResult=None) -> float:
# 		'''
# 		Calculate the uncertainty in the intensity of partial wave `waveName`
# 		'''
# 		if result is None:
# 			result = self.best

# 		couplings = np.array([result.couplings[i] for i in self._indicesCouplings[waveName]])
# 		jacobian = np.empty(couplings.size*2)
# 		jacobian[::2] = 2*math.real(couplings)
# 		jacobian[1::2] = 2*math.imag(couplings)
# 		cov = np.empty((jacobian.size, jacobian.size))
# 		indicesCov = self._indicesCouplingsCovariance[waveName]
# 		for iRank, iIndicesCov in enumerate(indicesCov):
# 			iReal = 2*iRank
# 			iImag = 2*iRank+1
# 			for jRank, jIndicesCov in enumerate(indicesCov):
# 				jReal = 2*jRank
# 				jImag = 2*jRank+1
# 				cov[iReal,jReal] = result.covMatrix[iIndicesCov[0],jIndicesCov[0]]
# 				cov[iReal,jImag] = result.covMatrix[iIndicesCov[0],jIndicesCov[1]]
# 				cov[iImag,jReal] = result.covMatrix[iIndicesCov[1],jIndicesCov[0]]
# 				cov[iImag,jImag] = result.covMatrix[iIndicesCov[1],jIndicesCov[1]]
# 		return math.sqrt( math.einsum('i,ij,j->', jacobian,cov,jacobian) )






# 	def getCouplingSpinDensityMatrix(self, waveNameA: str, waveNameB: str, result: FitResult=None) -> float:
# 		'''
# 		Calculate the spin-density matrix element for waveA and waveB for the coupling amplitues. The latter one is complex conjugated
# 		'''
# 		if result is None:
# 			result = self.best
# 		couplingsA = np.array([result.couplings[i] for i in self._indicesCouplings[waveNameA]])
# 		couplingsB = np.array([result.couplings[i] for i in self._indicesCouplings[waveNameB]])
# 		return np.sum(couplingsA*math.conjugate(couplingsB))
