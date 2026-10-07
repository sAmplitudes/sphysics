# pylint: disable=too-many-public-methods
from __future__ import annotations

import pathlib
import os
import numpy as np
import yaml

from ... import math
from ...utils import Logger
from .._model import PWAModel
from ._binning import Binning
from ._waveset import WaveSet

log = Logger('PWA')

def covIntensity(c: np.ndarray,covC: np.ndarray) -> tuple:
	""" Calculate the covariance matrix for the intensity.

	Args:
	    c(np.array): complex coupling constants
	    covC(np.array): covariance matrix for the coupling constants
	    The shape of covariance matrix should be `2*length(c) x 2*length(c)`
	    The format of covariance matrix should be (Cov[Re(c1),Re(c1)], Cov[Re(c1),Im(c1)], Cov[Re(c1),Re(c2)], ...
		                                     Cov[Im(c1),Re(c1)], Cov[Im(c1),Im(c1)], ...
											 .
											 .
											 .
											                                                                  )

	Returns:
	    tuple: A tuble (intensity, covI), where the intensity is an array of shape[c] and the covariance matrix of the intensity (covI) is an array of shape `length(c) x length(c)`)
	"""
	c = np.array(c)
	cReIm = np.empty(c.size*2, dtype=float)
	cReIm[::2] = c.real
	cReIm[1::2] = c.imag
	intensity = math.abs2(c)
	d = (len(c), len(cReIm))
	j = np.zeros(d)
	idxRow = np.arange(len(c))
	idxColumn = np.arange(len(c) * 2)
	idxColumn = idxColumn[idxColumn % 2 == 0]
	j[idxRow, idxColumn] = 2*cReIm[::2]
	j[idxRow, idxColumn + 1] = 2*cReIm[1::2]
	jTranspose = np.transpose(j)
	covJT = np.matmul(covC, jTranspose)
	covI = np.matmul(j, covJT)
	return intensity, covI


def covPhase(c: np.ndarray,covC: np.ndarray) -> tuple:
	""" Calculate the covariance matrix for the phase.

	Args:
	    c(np.array): complex coupling constants
	    covC(np.array): covariance matrix for the coupling constants
	    The shape of covariance matrix should be `2*length(c) x 2*length(c)`
	    The format of covariance matrix should be (Cov[Re(c1),Re(c1)], Cov[Re(c1),Im(c1)], Cov[Re(c1),Re(c2)], ...
		                                     Cov[Im(c1),Re(c1)], Cov[Im(c1),Im(c1)], ...
											 .
											 .
											 .
											                                                                  )

	Returns:
	    tuple: A tuble (phase, covPhi), where the phase is an array of shape [c] and the covariance matrix of the phase(covPhi) is an array of the shape `length(c) x length(c)`)
	"""
	c = np.array(c)
	cReIm = np.empty(c.size*2, dtype=float)
	cReIm[::2] = c.real
	cReIm[1::2] = c.imag
	if np.any((c.real==0) & (c.imag==0)):
		raise Exception("undefined phase value due to invalid entry: 0+0j not allowed as an input in the first parameter of the covPhase() function ")
	phase = np.angle(c, deg=False)
	d = (len(c), len(cReIm))
	j = np.zeros(d)
	idxRow = np.arange(len(c))
	idxColumn = np.arange(len(c) * 2)
	idxColumn = idxColumn[idxColumn % 2 == 0]
	denom = cReIm[::2]*cReIm[::2] + cReIm[1::2]*cReIm[1::2]
	j[idxRow, idxColumn] = (-1)*cReIm[1::2]/denom
	j[idxRow, idxColumn + 1] = cReIm[::2]/denom
	jTranspose = np.transpose(j)
	covJT = np.matmul(covC, jTranspose)
	covPhi = np.matmul(j, covJT)
	return phase, covPhi


def covBranchingRatio(c: np.ndarray, i: np.ndarray, covC: np.ndarray) -> tuple:
	""" Calculate the branching ratio and its covariance matrix.

	Args:
		c(np.array): complex coupling constants
		i(np.array): hermitian integral matrix, the shape of the integral matrix should be `length(c) x length(c)`
		covC(np.array): covariance matrix for the coupling constants, the shape of covariance matrix should be `2*length(c) x 2*length(c)`
		The format of covariance matrix should be (Cov[Re(c1),Re(c1)], Cov[Re(c1),Im(c1)], Cov[Re(c1),Re(c2)], ...
													Cov[Im(c1),Re(c1)], Cov[Im(c1),Im(c1)], ...
																													)
	Returns:
		tuple: A tuble (branchingRatio, covBR) where branchingRatio is an array of shape [c] and the covariance matrix of branching ratio(covBr)
		is an array of shape `length(c) x length(c)`
	"""
	iHermitian = np.matrix(i).getH()  # pylint: disable=assignment-from-no-return
	if np.any(i!= iHermitian):
		raise Exception("i must be hermitian")

	c = np.array(c)
	cReIm = np.empty(c.size*2, dtype=float)
	cReIm[::2] = c.real
	cReIm[1::2] = c.imag

	cConj = np.conjugate(c)
	inSum = np.outer(c, cConj)
	nInput = inSum*i
	n = np.sum(nInput)
	branchingRatio = math.abs2(c)/n
	d = (len(c), len(cReIm))
	j = np.zeros(d, dtype = "complex_")
	xReI = np.matmul(c.real, i)
	ixRe = np.matmul(i, c.real)
	xImI = np.matmul(c.imag*1j , i)
	ixIm = np.matmul(i , c.imag*1j)

	yImI = np.matmul(c.imag, i)
	iyIm = np.matmul(i, c.imag)
	yReI = np.matmul(c.real*1j , i)
	iyRe = np.matmul(i , c.real*1j)

	idx_row = np.arange(len(c))
	idx_column = np.arange(len(cReIm))
	idx_column = idx_column[idx_column % 2 == 0]
	j[idx_row, idx_column] = (2*cReIm[::2]-branchingRatio*(xReI+ixRe+xImI-ixIm))/n
	j[idx_row, idx_column + 1] = (2*cReIm[1::2]-branchingRatio*(yImI+iyIm+iyRe-yReI))/n

	jTranspose = np.transpose(j)
	covJT = np.matmul(covC, jTranspose)
	covBr = np.matmul(j, covJT)
	return branchingRatio, covBr


def covFitFraction(couplings: np.ndarray, integralMatrix: np.ndarray, isobarFraction: np.ndarray, covC: np.ndarray) -> tuple:
	""" Calculate the Fit fraction and its covariance matrix

	Args:
		couplings(np.array): complex coupling constants
		ntegralMatrix(np.array): hermitian integral matrix, the shape of the integral matrix should be len(couplingsxlen(couplings)
		isobarFraction(np.array): coefficients for isobar correction, the shape of covariance matrix should be 2*len(couplings)
		covC(np.array): covariance matrix for the coupling constants, the shape of isobarFraction should be len(couplings)
		The format of covariance matrix should be (Cov[Re(c1),Re(c1)], Cov[Re(c1),Im(c1)], Cov[Re(c1),Re(c2)], ...
													Cov[Im(c1),Re(c1)], Cov[Im(c1),Im(c1)], ...
																													)
	Returns:
		tuple: (Fit fraction:array of shape[c], covariance matrix of fit fraction:array of shape len(couplings)xlen(couplings),
				uncertainty of fit fraction:array of shape len(couplings))
	"""
	iHermitian = np.matrix(integralMatrix).getH()  # pylint: disable=assignment-from-no-return
	if np.any(integralMatrix!= iHermitian):
		raise Exception("integralMatrix must be hermitian")
	intensity = math.einsum('a,ab,b', couplings, integralMatrix, math.conjugate(couplings))
	fitFraction = math.abs2(couplings)/intensity
	fitFractionIsobar = fitFraction*isobarFraction
	couplingsReIm = np.empty(couplings.size*2, dtype=float)
	couplinsRealPart = couplings.real
	couplingsImaginaryPart = couplings.imag
	couplingsReIm[::2] = couplinsRealPart
	couplingsReIm[1::2] = couplingsImaginaryPart
	couplingsComplexLength = len(couplings)
	coplingsReImLength = len(couplingsReIm)
	jacobianDim = (couplingsComplexLength, coplingsReImLength)
	jacobianStart = np.zeros(jacobianDim, dtype = "complex_")
	idxRow = np.arange(couplingsComplexLength)
	idxColumn = np.arange(coplingsReImLength)
	idxColumn = idxColumn[idxColumn % 2 == 0]
	nomDxDy = jacobianStart
	nomDxDy[idxRow, idxColumn] = isobarFraction*(2*couplingsReIm[::2])/intensity
	nomDxDy[idxRow, idxColumn + 1] = isobarFraction*(2*couplingsReIm[1::2])/intensity
	integralMatrixT = np.transpose(integralMatrix)
	integralMatrixReT = integralMatrixT.real
	integralMatrixImT = integralMatrixT.imag
	denomDx = 2*(np.matmul(integralMatrixReT, couplinsRealPart) - np.matmul(integralMatrixImT, couplingsImaginaryPart))/intensity
	denomDy = 2*(np.matmul(integralMatrixReT, couplingsImaginaryPart) - np.matmul(integralMatrixImT , couplinsRealPart))/intensity
	jacobianPart1 = nomDxDy
	locDx = np.arange(1, couplingsComplexLength+1)
	jacobianPart2x = np.insert(np.outer(fitFraction, denomDx), obj=locDx, values=0, axis=1)
	locDy = np.arange(couplingsComplexLength)
	jacobianPart2y = np.insert(np.outer(fitFraction, denomDy), locDy, values=0, axis=1)
	jacobian = jacobianPart1 - jacobianPart2x - jacobianPart2y
	jacobianT = np.transpose(jacobian)
	covJT = np.matmul(covC, jacobianT)
	covFitFr = np.matmul(jacobian, covJT)
	if np.any(covFitFr.imag) != 0 and np.all(covFitFr.real/covFitFr.imag) > 1e-17:
		covFitFr = covFitFr.real
	uncertaintyFitFr = np.sqrt(np.diag(covFitFr))
	return fitFractionIsobar, covFitFr, uncertaintyFitFr




class PWAConfig:
	"""Class holding the configuration for the PWA."""
	def __init__(self, configFilePath: str|pathlib.Path):
		"""Initialize the PWA configuration.

		Args:
			configFilePath (str | pathlib.Path): Path to the configuration file
		"""
		with open(configFilePath, 'r', encoding='utf-8') as fin:
			self._config = yaml.load(fin, yaml.Loader)

		self.validate_required_entries()

		self._binning = Binning()
		binning_limits = {
			var: (binning['n_bins'], binning['lower'], binning['upper'])
			for var, binning in self._config['binning'].items()
		}
		self._binning.setLimits(binning_limits)

		self._configFilePath = pathlib.Path(configFilePath).resolve()


	def validate_required_entries(self):
		"""Validate required entries in the configuration."""
		required_keys = ['binning', 'paths']
		for key in required_keys:
			if key not in self._config:
				log.raiseException(ValueError, f"Required entry '{key}' is missing from the config file.")
		# Additional check for 'binning' structure
		if not isinstance(self._config['binning'], dict):
			log.raiseException(ValueError, "The 'binning' entry must be a dictionary.")
		for var, binning in self._config['binning'].items():
			if not isinstance(binning, dict) or not all(k in binning for k in ['n_bins', 'lower', 'upper']):
				log.raiseException(ValueError, f"Binning for variable '{var}' must contain 'n_bins', 'lower', and 'upper'.")
		# Additional check for 'paths' structure
		if not isinstance(self._config['paths'], dict):
			log.raiseException(ValueError, "The 'paths' entry must be a dictionary.")
		# Check for required paths
		required_paths = ['modelPath', 'waveset']
		for path_key in required_paths:
			if path_key not in self._config['paths']:
				log.raiseException(ValueError, f"Required path '{path_key}' is missing from the 'paths' entry in the config file.")
		# Check for required additional settings
		if 'numFitAttemptsPerThread' not in self._config:
			log.raiseException(ValueError, "Required entry 'numFitAttemptsPerThread' is missing from the config file.")

	def makePathsAbsolute(self):
		"""Make all paths in the configuration absolute paths, relative to the configuration file's parent directory.
		"""
		for key in self._config['paths']:
			self._config['paths'][key] = str(self.getPath(key))

	def setConfigFilePath(self, newConfigFilePath: str|pathlib.Path):
		"""Set a new path for the configuration file and update relative paths.

		Args:
			newConfigFilePath (str | pathlib.Path): The new path for the configuration file
		"""
		old_parent = pathlib.Path(self._configFilePath).parent
		new_parent = pathlib.Path(newConfigFilePath).resolve().parent
		self._configFilePath = pathlib.Path(newConfigFilePath).resolve()
		for key in self._config['paths']:
			old_path = pathlib.Path(self._config['paths'][key])
			if not old_path.is_absolute():
				# Resolve relative to old config file location, then make relative to new location
				absolute_path = (old_parent / old_path).resolve()
				self._config['paths'][key] = os.path.relpath(absolute_path, new_parent)


	@property
	def binning(self)->Binning:
		'''Returns the binning of the PWA model
		'''
		return self._binning

	@property
	def configFilePath(self) -> pathlib.Path:
		"""Return the configuration file path.

		Returns:
			pathlib.Path: The configuration file path
		"""
		return self._configFilePath

	@property
	def description(self) -> str:
		"""Return the description of the configuration.

		Returns:
			str: The description of the configuration
		"""
		if 'description' in self._config:
			return self._config['description']
		description = []
		description.append(f"PWD: {pathlib.Path(self.configFilePath).parent.name}")
		description.append(f"Model: {pathlib.Path(self.getModelPath()).name}")
		description.append(f"Waveset: {pathlib.Path(self.getWavesetPath()).name}")
		description.append(f"Fitted data: {pathlib.Path(self.getFittedDataDir()).name}")
		return ", ".join(description)


	def getAuxiliary(self, key: str):
		"""Retrieve auxiliary configuration data.

		Args:
			key (str): The key for the auxiliary data to retrieve

		Returns:
			Any: The auxiliary data associated with the given key
		"""
		return self._config[key]


	def getAuxiliaryDefault(self, key: str, defaultValue):
		"""Retrieve auxiliary configuration data with a default value.

		Args:
			key (str): The key for the auxiliary data to retrieve
			defaultValue: The default value to return if the key is not found

		Returns:
			Any: The auxiliary data associated with the given key, or the default value if the key is not found
		"""
		return self._config.get(key, defaultValue)


	def getPath(self, key: str, resolve: bool = True) -> pathlib.Path:
		"""Retrieve a resolved path from the configuration.

		Args:
			key (str): The configuration key identifying the desired path.
			resolve (bool): Whether to resolve the path to an absolute path. Defaults to True.

		Returns:
			pathlib.Path: The resolved absolute path corresponding to the given
				key, relative to the configuration file's parent directory.
			None: If the key does not exist in the configuration paths.
		"""
		if key not in self._config['paths']:
			return None
		path = self.configFilePath.parent / self._config['paths'][key]
		if resolve:
			path = path.resolve()
		return path



	def update(self, new_data: dict):
		"""Update the configuration with new data.

		Args:
			new_data (dict): New data to update the configuration with
		"""
		self._config.update(new_data)


	def dump(self, configFilePath: str|pathlib.Path = None):
		"""Dump the configuration to a file.

		Args:
			configFilePath (str | pathlib.Path): Path to the configuration file, if None, use the original path
		"""
		if configFilePath is None:
			configFilePath = self.configFilePath

		with open(configFilePath, 'w', encoding='utf-8') as fin:
			yaml.dump(self._config, fin, default_flow_style=False)


	def getBinFittedDataPath(self, binNumber: int = None) -> str:
		"""Get Path to a bin of fitted Data (data that will be fit in the PWD).

		Args:
			binNumber (int, optional): The bin number to get the path for. Defaults to None.

		Returns:
			str: Path to the bin of fitted data.
		"""
		basePath = self.getPath("binFittedData")
		if basePath is None:
			basePath = (self.configFilePath.parent / 'binFittedData').resolve()
		if binNumber is None:
			return str(basePath)
		binNumber = self.binning.validateBin(binNumber)
		return str(basePath / f'bin{binNumber}' / f'merged_binFittedData_bin{binNumber}.hdf5')


	def getBinGenMCDataPath(self, binNumber: int = None) -> str:
		"""Get Path to a bin of genMC Data, i.e. the data that we use to calculate the genMC integral matrix.


		Args:
			binNumber (int, optional): The bin number to get the path for. Defaults to None.

		Returns:
			str: Path to the bin of genMC data.
		"""
		basePath = self.getPath("binGenMCData")
		if self.getAuxiliary('paths').get('binGenMCData') is None:
			basePath = (self.configFilePath.parent / 'binGenMCData').resolve()
		if binNumber is None:
			return str(basePath)
		binNumber = self.binning.validateBin(binNumber)
		return str(basePath / f'bin{binNumber}' / f'merged_binGenMCData_bin{binNumber}.hdf5')


	def getBinReconMCDataPath(self, binNumber: int = None) -> str:
		"""Get Path to a bin of reconMC Data, i.e. the data that we use to calculate the reconMC integral matrix.


		Args:
			binNumber (int, optional): The bin number to get the path for. Defaults to None.

		Returns:
			str: Path to the bin of reconMC data.
		"""
		basePath = self.getPath("binReconMCData")
		if self.getAuxiliary('paths').get('binReconMCData') is None:
			basePath = (self.configFilePath.parent / 'binReconMCData').resolve()
		if binNumber is None:
			return str(basePath)
		binNumber = self.binning.validateBin(binNumber)
		return str(basePath / f'bin{binNumber}' / f'merged_binReconMCData_bin{binNumber}.hdf5')


	def getBinnedDataDir(self, dataType: str, binNumber: int = None) -> str:
		"""Get Path to directory of fitted/genMC/reconMC data.

		Args:
			dataType (str): Type of data. Options are 'fittedDataDir', 'genMCDataDir', 'reconMCDataDir'
			binNumber (int, optional): The bin number to get the path for. Defaults to None.

		Returns:
			str: Path to the binned data directory.
		"""
		if dataType not in ['binFittedData', 'binGenMCData', 'binReconMCData']:
			raise Exception(f'Invalid dataType: {dataType}. Options are "fittedDataDir", "genMCDataDir", "reconMCDataDir".')
		if dataType == 'binFittedData':
			return self.getBinFittedDataPath(binNumber)
		if dataType == 'binGenMCData':
			return self.getBinGenMCDataPath(binNumber)
		return self.getBinReconMCDataPath(binNumber)


	def getFittedDataDir(self) -> str:
		"""Get Path to fitted Data Dir, i.e. the data that will be fit in the PWD.

		Returns:
			str: Path to the fitted data directory.
		"""
		basePath = self.getPath("fittedDataDir")
		if basePath is None:
			raise Exception("The path to the fitted data directory is not specified in the configuration file under the keys 'paths' -> 'fittedDataDir'.")
		return basePath


	def getGenMCDataDir(self) -> str:
		"""Get Path to a bin of genMC Data, i.e. the data that we use to calculate the genMC integral matrix.

		Returns:
			str: Path to the genMC data directory.
		"""
		basePath = self.getPath("genMCDataDir")
		if basePath is None:
			raise Exception("The path to the genMC data directory is not specified in the configuration file under the keys 'paths' -> 'genMCDataDir'.")
		return basePath


	def getReconMCDataDir(self) -> str:
		"""Get Path to a bin of reconMC Data, i.e. the data that we use to calculate the reconMC integral matrix.

		Returns:
			str: Path to the reconMC data directory.
		"""
		basePath = self.getPath("reconMCDataDir")
		if basePath is None:
			raise Exception("The path to the reconMC data directory is not specified in the configuration file under the keys 'paths' -> 'reconMCDataDir'.")
		return basePath


	def getModelPath(self) -> str:
		"""Get Path to the Fitting model

		Returns:
			str: Path to the fitting model
		"""
		return str(self.getPath("modelPath"))


	def getDecayAmplitudesPath(self, binNumber: int) -> str:
		"""Get Path to a bin of the decay amplitudes.

		Args:
			binNumber (int): The bin number to get the path for.

		Returns:
			str: Path to the bin of the decay Amplitudes.
		"""
		binNumber = self.binning.validateBin(binNumber)
		basePath = self.getPath("decayAmplitudes")
		if basePath is None:
			basePath = self.configFilePath.parent / 'decayAmplitudes'
		return str((basePath / f'bin{binNumber}' / f'decayAmplitudes_bin{binNumber}.npy').resolve())


	def getIntMatsGenMCPath(self, binNumber: int = None) -> str:
		"""Get Path to the merged genMC integral matrices or the binned genMC integral matrix.


		Args:
			binNumber (int, optional): The bin number to get the path for. Defaults to None.

		Returns:
			str: Path to the genMC integral matrices.
		"""
		basePath = self.getPath("intMatsGenMC")
		if basePath is None:
			basePath = (self.configFilePath.parent / 'intMatsGenMC').resolve()
		else:
			basePath = basePath.resolve()
		if binNumber is None:
			path = str(basePath / 'allGenIntMat.npy')
		else:
			binNumber = self.binning.validateBin(binNumber)
			path = str(basePath / f'bin{binNumber}' / f'genMCIntMat_bin{binNumber}.npy')
		return path


	def getIntMatsReconMCPath(self, binNumber: int = None) -> str:
		"""Get Path to the merged reconMC integral matrices or the binned reconMC integral matrix.


		Args:
			binNumber (int, optional): The bin number to get the path for. Defaults to None.

		Returns:
			str: Path to the reconMC integral matrices.
		"""
		basePath = self.getPath("intMatsReconMC")
		if basePath is None:
			basePath = (self.configFilePath.parent / 'intMatsReconMC').resolve()
		else:
			basePath = basePath.resolve()
		if binNumber is None:
			path = str(basePath / 'allReconIntMat.npy')
		else:
			binNumber = self.binning.validateBin(binNumber)
			path = str(basePath / f'bin{binNumber}' / f'reconMCIntMat_bin{binNumber}.npy')
		return path


	def getNormIntPath(self, binNumber: int = None) -> str:
		"""Get Path to the merged normalization integrals or the binned normalization integrals.


		Args:
			binNumber (int, optional): The bin number to get the path for. Defaults to None.

		Returns:
			str: Path to the normalization integrals.
		"""
		basePath = self.getPath("normInt")
		if basePath is None:
			basePath = (self.configFilePath.parent / 'normInt').resolve()
		else:
			basePath = basePath.resolve()
		if binNumber is None:
			path = str(basePath / 'allNormInt.pkl')
		else:
			binNumber = self.binning.validateBin(binNumber)
			path = str(basePath / f'bin{binNumber}' / f'normIntArrays_bin{binNumber}.npy')
		return path


	def getNormIntNumpyPath(self) -> str:
		"""Get Path to allNormInt.npy.

		Returns:
			str: Path to allNormInt.npy.
		"""
		basePath = self.getPath("normInt")
		if basePath is None:
			basePath = self.configFilePath.parent / 'normInt'
		return str((basePath / 'allNormInt.npy').resolve())


	def getNormIntArraysPath(self) -> str:
		"""Get Path to allNormIntArrays.npy.

		Returns:
			str: Path to allNormIntArrays.npy.
		"""
		basePath = self.getPath("normInt")
		if basePath is None:
			basePath = self.configFilePath.parent / 'normInt'
		return str((basePath / 'allNormIntArrays.npy').resolve())


	def getPlotCollectionPath(self) -> str:
		"""Get Path to the PlotCollection.

		Returns:
			str: Path to PlotCollection.
		"""
		return str((self.configFilePath.parent / 'plotCollection' / 'plotCollection.root').resolve())


	def getInputModelCurvePlotCollectionPath(self) -> str:
		"""Get Path to the ModelCurvePlotCollection.

		Returns:
			str: Path to ModelCurvePlotCollection.
		"""
		if self.getPath("inputModelCurvePlotCollection") is not None:
			return str(self.getPath("inputModelCurvePlotCollection"))
		return str((self.configFilePath.parent / 'plotCollection' / 'modelCurvePlotCollection.root').resolve())


	def getFitResultsPath(self, binNumber: int = None, fitThreadNum: int = None) -> str:
		"""Get Path to the merged fit results, the binned fit results and the binned fit results in their separet threads.

		Args:
			binNumber (int, optional): The bin number of the wanted fit result. Defaults to None.
			fitThreadNum (int, optional): Thread number of the wanted fit result in a certain bin. Defaults to None.

		Returns:
			str: Path to the fit results.
		"""
		if binNumber is not None:
			binNumber = self.binning.validateBin(binNumber)
			if fitThreadNum is not None:
				if self._config.get('numFitThreads') is None:
					raise Exception("The number of fit threads is not specified in the configuration file. Probably the fit was not run with multiple threads. " \
					"Only use binNumber argument to get the fit results for a specific bin.")
				if fitThreadNum<0 or fitThreadNum>=self.getAuxiliary('numFitThreads'):
					raise Exception(f'The requested fit thread number {fitThreadNum} is out of range. Valid fit thread numbers are between 0 and {self.getAuxiliary("numFitThreads")-1}.')
				path = str((self.configFilePath.parent / 'fitResults' / f'bin{binNumber}' / f'fitResults_bin{binNumber}_thread{fitThreadNum}.pkl').resolve())
			else:
				path = str((self.configFilePath.parent / 'fitResults' / f'bin{binNumber}' / f'fitResults_bin{binNumber}.pkl').resolve())
		else:
			path = str((self.configFilePath.parent / 'fitResults' / 'fitResults.pkl').resolve())
		return path


	def getBkgConfigFilePath(self) -> str:
		"""Get Path to the background config file.

		Returns:
			str: Path to background config file.
		"""
		if self.getPath("background") is not None:
			return str(self.getPath("background"))
		return str((self.configFilePath.parent / 'configBkg.yaml').resolve())


	def getPredictedBkgPath(self, binNumber: int = None) -> str:
		"""Get Path to the predicted background directory.

		Args:
			binNumber (int, optional): The bin number to get the path for.
				Defaults to None.

		Returns:
			str: Path to the predicted background directory or file.
		"""
		basePath = self.getPath("predictedBackground")
		if basePath is None:
			basePath = self.configFilePath.parent / 'predictedBackground'

		if binNumber is None:
			return str(basePath)

		binNumber = self.binning.validateBin(binNumber)
		return str((basePath / f'bin{binNumber}').resolve())

	def getWavesetPath(self) -> str:
		"""Get Path to the waveset config file.

		Returns:
			str: Path to waveset config file.

		Raises:
			ValueError: If the waveset path is not specified in the
				configuration file.
		"""
		if self.getPath("waveset") is None:
			raise Exception("The path to the waveset config file is not specified in the configuration file under the keys 'paths' -> 'waveset'.")
		return str(self.getPath("waveset"))

	def getWaveset(self) -> WaveSet:
		"""Get the waveset defined in the waveset file.

		Returns:
			WaveSet: The waveset defined in the waveset file.
		"""
		wavesetPath = self.getWavesetPath()
		return WaveSet.fromYaml(wavesetPath)

	def getModel(self) -> PWAModel:
		"""Get the model defined in the model file.

		Returns:
			PWAModel: The model function defined in the model file.
		"""
		modelPath = self.getModelPath()
		exec_namespace = {}
		with open(modelPath, 'r', encoding='utf-8') as file:
			model_code = file.read()
		#execute file contents:
		exec(model_code, exec_namespace) # pylint: disable=exec-used
		model = exec_namespace.get('model')
		return model

	def getNumFitAttemptsPerThread(self) -> int:
		"""Get the number of fit attempts per thread.

		Returns:
			int: The number of fit attempts per thread.

		Raises:
			ValueError: If the number of fit attempts per thread is not specified in the configuration file.
		"""
		return self.getAuxiliary('numFitAttemptsPerThread')



def constOne(x) -> float: # pylint: disable=unused-argument
	"""Return one for any input.

	Args:
		x (any): Can be any input

	Returns:
		float: 1.0
	"""
	return 1.
