# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Generic fitter for an unbinned extended maximum-likelihood PWA fit, Created on Tuesday 29 03 2022
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import iminuit
import tensorflow as tf
from tensorflow import math

from .._model import PWAModel
from ...utils import Logger

from ._startparameterGenrators import StartparameterGenerator, UniformStartparameterGenerator
from ._fitresult import createFitresult, ResultCollection

log = Logger('PWAFit')
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(5)

class ParaemterMapping:

	def __init__(self, model: PWAModel, nBackground: int):
		self._model = model
		self._nBkg = nBackground

	def idxWaveRealPart(self, waveName: str) -> int:
		"""Get index of parameter representing real part of wave
		"""
		return 2*self._model.getWaveIndex(waveName)

	def idxWaveImagPart(self, waveName: str) -> int:
		"""Get index of parameter representing imaginary part of wave
		"""
		return 2*self._model.getWaveIndex(waveName) + 1

	def n_param_couplings(self) -> int:
		return 2*self._model.nWaves

	def idxBackgroundParameters(self):
		start = self.n_param_couplings()
		end = start + self._nBkg
		return list(range(start, end))

	def n_param_total(self) -> int:
		return 2*self._model.nWaves + self._nBkg

	def physicsParameters2FitterParameters(self, couplings: np.ndarray) -> np.ndarray:
		'''Returns a 1D parameter array with 2 entries per coupling (one real and one imaginary parameter for each partial wave)
		based on the number of couplings of the partial wave model.

		:param couplings: Couplings of the partial waves
		:type couplings: np.ndarray
		:return: Fit parameters
		:rtype: np.ndarray
		'''
		nParameter = couplings.size*2
		parameters = np.empty(nParameter)
		parameters[::2] = math.real(couplings)
		parameters[1::2] = math.imag(couplings)
		return parameters

	@tf.function
	def fitterParameters2Couplings(self, parameters: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
		'''Converts a 1D parameter tensor into complex couplings (in the form of (real + j* imaginary)) and background yields.

		:param parameters: A tensor of shape (..., 2 * nWaves + nBkg) containing real-valued fit parameters. The first 2 * nWaves entries represent the
		real and imaginary parts of the complex couplings, followed by nBkg background yield parameters.
		:type parameters: tf.Tensor
		:return: Complex-valued tensor of shape (..., nWaves) representing the fitted couplings and real-valued tensor of shape (..., nBkg)
		containing background yield parameters.
		If `nBkg == 0`, this will be an empty tensor of shape (0,).
		:rtype: tuple[tf.Tensor, tf.Tensor]
		'''
		# parameters: (..., 2*nWaves + nBkg), float
		nCouplings = self.n_param_couplings()
		couplings = tf.dtypes.complex(parameters[:nCouplings:2], parameters[1:nCouplings:2])
		if self._nBkg > 0:
			backgroundYields = parameters[nCouplings:nCouplings+self._nBkg]
		else:
			backgroundYields = tf.zeros([0], dtype=parameters.dtype)
		return couplings, backgroundYields


	def intensitiesAndPhases2FitterParameters(self, intensities, phases) -> np.ndarray:
		"""
		Returns a 1D parameter array with 2 entries per coupling (one real and one imaginary parameter for each partial wave)
		calculated from the intensities and phases of each partial wave.

		:param intensities: Wave intensities
		:param phases: Phases
		:return: Returns a 1D parameter array with 2 entries per coupling
		:rtype: tuple[Tensor, Tensor]
		"""
		real = math.sqrt(intensities)* math.cos(phases)
		imag = math.sqrt(intensities)* math.sin(phases)

		params = np.empty(self._model.nWaves * 2)
		params[::2]  = real
		params[1::2] = imag

		return params

class Fitter():
	'''
	Generic fitter class for an unbinned extended maximum-likelihood PWA fit

	:param dataAmplitudes: Decay amplitudes of the data events that are fitted, shape (nWaves, nIncoherentSectors, nEvents)
	:param integralMatrix: Accepted phase-space integral over the outer-product of the decay amplitudes, incoherently summed over all incoherent indices of the shape (nWaves, nWaves)
	'''
	def __init__(self, model: PWAModel, dataDecayAmplitudes: np.ndarray, integralMatrixRecon: np.ndarray, integralMatrixGen: np.ndarray,
	             refWave: str, eventWeights: np.ndarray = None, startparameterGenerator: StartparameterGenerator = None,
				 tol: float = 1e-7, zeroWaves: list[str] = None,
				 backgroundTypes: list[str] | None = None, bFunction: tf.Tensor = None, fixedBackground = None, cauchyMulti: float = None) -> None:
		"""
		Initializes the fitter object for Partial Wave Analysis (PWA).
		Args:
			model (PWAModel): The PWA model instance containing wave definitions and related methods.
			dataDecayAmplitudes (np.ndarray): Array of decay amplitudes for the data events.
			integralMatrixRecon (np.ndarray): Matrix of integrals used in the likelihood calculation.
			integralMatrixGen (np.ndarray): Matrix of integrals used for calculations on generator level.
			refWave (str): Name of the reference wave, which is used as a phase reference and cannot be set to zero.
			eventWeights (np.ndarray, optional): Give weights for each event, needed e.g. for sideband subtraction.
			startparameterGenerator (StartparameterGenerator, optional): Generator for initial fit parameters.
				If None, a uniform generator is used by default.
			tol (float, optional): Tolerance for numerical operations, defaults to 1e-7.
			zeroWaves (list[str], optional): List of wave names to be fixed to zero during the fit.
				The reference wave cannot be included.
			backgroundTypes (list[str], optional): List of strings, each entry corresponding to a type of background. Defaults to None.
			bFunction (tf.Tensor, optional): Background function values. Defaults to None.
			cauchyMulti (float, optional): Give value of cauchy multiplier for wave selection method.
		Raises:
			ValueError: If the reference wave is included in the zeroWaves list.
		"""

		self._model = model
		self._dataDecayAmplitudes = tf.constant(dataDecayAmplitudes)
		self._integralMatrixRecon = tf.constant(integralMatrixRecon)
		self._integralMatrixGen = tf.constant(integralMatrixGen)
		self._nEvents = self._dataDecayAmplitudes.shape[-1]
		self._eventWeights = tf.ones(self._nEvents,tf.dtypes.float64) if eventWeights is None else tf.constant(eventWeights,tf.dtypes.float64)
		self._nWaves = model.nWaves
		self._backgroundTypes = backgroundTypes if backgroundTypes is not None else [] # argument "backgroundTypes" passed from b2Luigi is a python list
		self._nBkg = len(self._backgroundTypes)
		self._bFunction = bFunction #should be shape(nBkg, nEvents)
		if bFunction is not None:
			self._bkgnEvents = bFunction.shape[1]
		else:
			self._bkgnEvents = 0

		self._parameterMapping = ParaemterMapping(model=self._model, nBackground=self._nBkg)
		self._nParameter = self._parameterMapping.n_param_total()
		self._n_ParameterCouplings = self._parameterMapping.n_param_couplings()

		self._refWaveIdx = self._model.getWaveIndex(refWave)
		self._refWaveRealParameterIdx = self._parameterMapping.idxWaveRealPart(refWave)
		self._refWaveImagParameterIdx = self._parameterMapping.idxWaveImagPart(refWave)

		# Dictionary containing <parameterindex>: <value> for parameters that should be fixed to the given value
		self._fixedParameters = {}
		self._fixedParameters[self._refWaveImagParameterIdx] = 0.0
		if fixedBackground is not None:
			for idx, y in fixedBackground:
				bkg_idx = self._parameterMapping.idxBackgroundParameters()[idx]
				self._fixedParameters[bkg_idx] = float(y)
		if zeroWaves is not None:
			for zeroWave in zeroWaves:
				if zeroWave == refWave:
					log.raiseException(ValueError, f"The reference wave '{refWave}' cannot be set to zero.")
				self._fixedParameters[self._parameterMapping.idxWaveRealPart(zeroWave)] = 0.0
				self._fixedParameters[self._parameterMapping.idxWaveImagPart(zeroWave)] = 0.0

		if startparameterGenerator is None:
			self._startparameterGenerator = UniformStartparameterGenerator(self._nWaves, self._refWaveIdx, self._nEvents, self._nBkg, self._bkgnEvents)
		else:
			self._startparameterGenerator = startparameterGenerator

		self._negLogLikelihoodWrapperTf_concrete = None
		self._gradient_concrete = None
		self._hesse_concrete = None
		self._cauchyMulti = cauchyMulti
		self._cauchyMultiSquared = cauchyMulti**2 if cauchyMulti is not None else None

		self._tol = tol


	@tf.function
	def _calcNormIntegral(self, couplings: tf.Tensor, integralMatrix: tf.Tensor) -> tf.Tensor:
		'''
		Calculates the normalization integral of the model, i.e. the predicted number of observed events
		@param couplings: Complex-valued coupling amplitudes of the shape (nWaves)
		@param integralMatrix: Accepted phase-space integral over the outer-product of the decay amplitudes, incoherently summed over all incoherent indices of the shape (nWaves, nWaves)
		'''
		return math.real(tf.einsum('a,b,ab', couplings, math.conj(couplings), integralMatrix))

	@tf.function
	def _calcCauchyTerm(self, couplings: tf.Tensor) -> tf.Tensor:
		'''
		Calculates the cauchy term of the couplings, i.e. the predicted number of observed events
		@param couplings: Complex-valued coupling amplitudes of the shape (nWaves)
		'''
		return math.reduce_sum(math.log(math.add(tf.ones(couplings.shape,tf.dtypes.float64),math.abs(math.multiply(math.abs(couplings)**2,1./self._cauchyMultiSquared)))),axis=0)

	@tf.function
	def _negLogLike(self, couplings: tf.Tensor, dataDecayAmplitudes: tf.Tensor, integralMatrix: tf.Tensor, eventWeights: tf.Tensor = None,
				 backgroundYields: tf.Tensor = None, bFunction: tf.Tensor = None) -> tf.Tensor:
		'''
		Basic negative log-likelihood function using the natural logarithm
		@param couplings: Complex-valued coupling amplitudes of the shape (nWaves)
		@param dataAmplitudes: Decay amplitudes of the data events that are fitted, shape (nWaves, nIncoherentSectors, nEvents)
		@param integralMatrix: Accepted phase-space integral over the outer-product of the decay amplitudes, incoherently summed over all incoherent indices of the shape (nWaves, nWaves)
		'''
		if self._nBkg > 0  and bFunction is not None:
			negLogLike = self._calcNormIntegral(couplings, integralMatrix) \
			+ tf.reduce_sum(backgroundYields) \
			- tf.tensordot(math.log(math.reduce_sum(math.abs(math.reduce_sum(couplings[:,None,None]*dataDecayAmplitudes, axis=0))**2, axis=0)
			+ tf.tensordot(backgroundYields, bFunction, axes=1)),eventWeights,axes=1)
		else:
			negLogLike = self._calcNormIntegral(couplings, integralMatrix) \
			- tf.tensordot(math.log(math.reduce_sum(math.abs(math.reduce_sum(couplings[:,None,None]*dataDecayAmplitudes, axis=0))**2, axis=0)),eventWeights,axes=1)
		#     sum over events         sum over incoh. sectors        sum over waves
		if self._cauchyMulti is not None:
			negLogLike += self._calcCauchyTerm(couplings)
		return negLogLike


	@tf.function
	def _negLogLikelihoodWrapper(self, parameters: tf.Tensor) -> tf.Tensor:
		couplings, backgroundYields = self._parameterMapping.fitterParameters2Couplings(parameters)
		return self._negLogLike(couplings, self._dataDecayAmplitudes, self._integralMatrixRecon, self._eventWeights, backgroundYields, self._bFunction)

	@tf.function
	def _gradient(self, parameters: tf.Tensor) -> tf.Tensor:
		with tf.GradientTape() as tape:
			value = self._negLogLikelihoodWrapper(parameters)
		return tape.gradient(value, parameters)


	@tf.function
	def _hesse(self, parameters: tf.Tensor) -> tf.Tensor:
		with tf.GradientTape() as tape1:
			with tf.GradientTape() as tape2:
				value = self._negLogLikelihoodWrapper(parameters)
			gradient = tape2.gradient(value, parameters)
		hesse = tape1.jacobian(gradient, parameters)
		return hesse

	def _minimizerFunction(self, parameters: np.ndarray) -> float:
		return self._negLogLikelihoodWrapperTf_concrete(tf.Variable(parameters, dtype=tf.float64)).numpy()

	def _minimizerGradient(self, parameters: np.ndarray) -> np.ndarray:
		return self._gradient_concrete(tf.Variable(parameters, dtype=tf.float64)).numpy()

	def _minimizerHesse(self, parameters: np.ndarray) -> np.ndarray:
		return self._hesse_concrete(tf.Variable(parameters, dtype=tf.float64)).numpy()

	def fit(self, nAttempts: int, tol: float=None) -> ResultCollection:
		'''
		Perform fit of `nAttempts` fit attempts with different start-parameter values

		:param tol: See iminuit tol definition
		'''

		if tol is None:
			tol = self._tol

		#####################
		# construct functions
		dummyParameters = tf.Variable(np.arange(self._nParameter, dtype=float))
		self._negLogLikelihoodWrapperTf_concrete = self._negLogLikelihoodWrapper.get_concrete_function(dummyParameters) # pylint: disable=no-member
		self._gradient_concrete = self._gradient.get_concrete_function(dummyParameters) # pylint: disable=no-member
		self._hesse_concrete = self._hesse.get_concrete_function(dummyParameters) # pylint: disable=no-member

		results = ResultCollection(model=self._model,
								   indicesCouplings={waveName: (i,) for i, waveName in enumerate(self._model.waveNames)},
		                           indicesCouplingsCovariance=[ (2*i, 2*i+1) for i, _ in enumerate(self._model.waveNames)],
								   integralMatrixRecon=self._integralMatrixRecon,
								   integralMatrixGen=self._integralMatrixGen
								   )
		for iAttempt in range(nAttempts):
			log.info(f"Start fit-attempt {iAttempt}")
			log.incrementIndent()
			# startCouplings = self._startparameterGenerator.generateCouplings()
			# startParameters = self._parameterMapping.physicsParameters2FitterParameters(startCouplings)
			startIntensities, startPhases = self._startparameterGenerator.generateIntensitiesandPhases()
			startParameters = self._parameterMapping.intensitiesAndPhases2FitterParameters(startIntensities, startPhases)
			if self._nBkg > 0:
				startBackgroundYields = self._startparameterGenerator.generateBackgroundYields()
				startParameters = np.concatenate([startParameters, startBackgroundYields])
			minuit = iminuit.Minuit(self._minimizerFunction,
			                        startParameters,
									grad=self._minimizerGradient
									)
			minuit.errordef = iminuit.Minuit.LIKELIHOOD
			minuit.tol = tol
			minuit.limits[f'x{self._refWaveRealParameterIdx}'] = (0.0, None)

			for idx in self._parameterMapping.idxBackgroundParameters():
				minuit.limits[f'x{idx}'] = (-1000.0, None)

			for iPara, value in self._fixedParameters.items():
				minuit.values[f'x{iPara}'] = value
				minuit.fixed[f'x{iPara}'] = True
			minuit.migrad()
			minuit.hesse()
			if minuit.fmin.has_accurate_covar or minuit.fmin.has_made_posdef_covar:
				covMatrix = np.array(minuit.covariance)
			else:
				covMatrix = None
			covMatrixValide = minuit.fmin.has_accurate_covar
			covMatrixMadePosDef = minuit.fmin.has_made_posdef_covar


			params_tf = tf.constant([p.value for p in minuit.params], dtype=tf.float64)
			couplings_tf, backgroundYields_tf = self._parameterMapping.fitterParameters2Couplings(params_tf)
			couplings = couplings_tf.numpy()
			backgroundYields = backgroundYields_tf.numpy()
			# remove the minimizer function from minuit as it is not pickable
			minuit._fcn = None # pylint: disable=protected-access
			fitResult = createFitresult(
				                  couplings=couplings,
								  covarianceMatrix=covMatrix,
								  auxiliaryParameters=backgroundYields,
								  auxiliaryParameterNames=self._backgroundTypes,
			                      negLogLikelihood=minuit.fmin.fval,
								  fitConverged=minuit.fmin.is_valid,
								  covMatrixValide=covMatrixValide,
								  covMatrixMadePosDef=covMatrixMadePosDef,
								  resultObj=minuit)
			results.insert(fitResult)
			log.decrementIndent()

		return results


	def _calculateAndCheckCovarianceMatrix(self, minuit: iminuit.Minuit, verbose: bool = False) -> tuple[np.ndarray, bool]:
		"""Calculate and check the covariance matrix

		- Calculates the hesse matrix at the fit result stored in `minuit` using auto-differenciation
		- Removes fixed parameters, in order to invert the hesse matrix
		- Inverts the hesse matrix -> covariance matrix
		- Checks positive definiteness of the covariance matrix (strictly speaking of the sub-matrix of not-fixed parameters)
		- Obtain full covariance matrix including fixed parameters with zeros


		Args:
			minuit (iminuit.Minuit): Minimizer object, which holds the parameters at which the covariance matrix will be calculated
			verbose (bool, optional): Print additional information of the covariance matrix is not positive devinit. Defaults to False.

		Returns:
			covarianceMatrix (np.ndarray): Covariance matrix (including fixed parameters).
			                               The order is the same as for the parameter list.
			covarianceMatrixValid( bool): True if the covariance matrix is valid, i.e. positive devinit
		"""

		hesseMat = self._minimizerHesse([p.value for p in minuit.params])

		indicesFree = np.array([i for i in range(hesseMat.shape[0]) if i != self._refWaveImagParameterIdx]).reshape(-1,1)

		hesseMatSub = hesseMat[indicesFree,indicesFree.T]

		covSub = np.linalg.inv(hesseMatSub)

		covValid = True
		evals, _ = np.linalg.eig(covSub)
		if np.sum(evals <= 0.0) > 0:
			covValid=False
			if verbose:
				log.warning("Covariance matrix is not positive definite with {0} non-positive eigenvalue(s):", np.sum(evals <= 0.0))
			log.incrementIndent()
			for eigenvalue in filter(lambda e: e <=0.0, evals):
				if verbose:
					log.warning(eigenvalue)
			log.decrementIndent()

		cov = np.zeros_like(hesseMat)
		cov[indicesFree, indicesFree.T] = covSub

		return cov, covValid
