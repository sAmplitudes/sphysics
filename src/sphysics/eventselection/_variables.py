# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Class holding various variables from various channels, Created on Thursday 10 03 2022
'''
#pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import os
from typing import Sequence
import pickle
import copy
import gzip
import functools
import collections
import numpy as np
import h5py
import pandas as pd
import polars as pl
import uproot

from .. import random
from .. import math
from ..utils import Logger

log = Logger("eventselection")



NUM_UPROOT_THREADS = 4 # small test shows that there is not big improvement beyond 4 threads

MOMENTUM_COMPONENTS = ['E', 'px', 'py', 'pz']


def _moveColumnToNumpy(dataFrame: pl.DataFrame, columnName: str, zeroCopy: bool) -> np.ndarray:
	'''Moves a single column out of `dataFrame` into a numpy array.

	The column is detached from the data frame before the conversion, so that at most one
	single column, and never the full data frame, is held twice in memory.

	:param dataFrame: Data frame from which the column is removed. CHANGES THE INPUT DATAFRAME.
	:param columnName: Name of the column to move
	:param zeroCopy: Share the Arrow buffer instead of copying it. The returned array is then READ-ONLY.
	:return: Array with the column data
	'''
	series = dataFrame.drop_in_place(columnName)
	if series.n_chunks() > 1: # a chunked column has no contiguous buffer to share
		series = series.rechunk()
	if zeroCopy:
		try:
			return series.to_numpy(allow_copy=False)
		except RuntimeError:
			pass # booleans, nulls and nested types have no numpy-compatible Arrow layout
	return series.to_numpy(writable=True)


class VariablesBase:
	"""Stores variables of a data sample

	Each member variable of this class is a variable.
	A variable can either be an array with an entry along the last axis per element in the sample, or a scalar variable.
	Data samples can be merged using the `self += other` operator.

		- Array variables are concatenated
		- Scalar variables of bot `Variables` objects have to be equal,
		- Except for dicts, which are merged, i.e. all keys of `other` missing in `self` are added
		  to self and the values of all keys in `self` and `other` have to be equal.

	"""
	def __init__(self) -> None:
		pass

	def hasVariable(self, variable: str) -> bool:
		'''Checks if a specified variable is in the current object.

		:param variable: The name of the variable
		:return: True, if variable exists, else False
		:rtype: bool
		'''
		return variable in self.getVariables()

	def __contains__(self, variable: str) -> bool:
		return self.hasVariable(variable)

	def isScalar(self, variable: str, _checkHasVariable: bool = True) -> bool:
		'''Checks if a specified variable is a scalar

		:param variable: The name of the variable
		:return: True, if variable is scalar, else False
		'''
		if _checkHasVariable and not self.hasVariable(variable):
			return False
		values = getattr(self, variable)
		if values is None:
			return False
		return np.isscalar(values) or isinstance(values, (dict,))

	def getVariables(self) -> list:
		'''Retrieves a sorted list of variables excluding \\:

		- Private / Special attributes
		- Properties
		- Callable attributes
		'''
		return sorted([ attr for attr in dir(self) if not attr.startswith('_') \
                                      and not isinstance(getattr(type(self), attr, None), property) \
                                                                    and not callable(getattr(self, attr))  \
                              ])

	def getVariableInfo(self, name: str) -> dict:
		'''Gets all the variable information for `addVariable`

		:param name: Name of variable
		:return: Dictionary containing variable information
		'''
		if not self.hasVariable(name):
			log.raiseException(Exception, f"Variables does not have variable `{name}`!")
		return {'name': name}

	def addVariable(self, name):
		'''Adds a new variable

		:param name: Name of variable
		:return: True, if variable is added, False if variable already exists
		'''
		if not self.hasVariable(name):
			setattr(self, name, None)
			return True
		print(f"Already added variable {name}!")
		return False

	def copy(self, doNotCopyNonscalarVariables=False):
		'''Creates a deep copy of the object.

		:param doNotCopyNonscalarVariables: Sets whether non-scalar variables should be copied. Defaults to False
		:return: A copy of the object, with or without non-scalar variables
		'''
		memory = {}
		for variable in self.getLoadedNonscalarVariables():
			memory[variable] = self[variable]
			self[variable] = None

		other = copy.deepcopy(self)
		for variable in memory:
			self[variable] = memory[variable]
			if not doNotCopyNonscalarVariables:
				other[variable] = memory[variable].copy()
		return other

	def __getitem__(self, key):
		if not self.hasVariable(key):
			raise Exception(f"Variable '{key}' not stored")
		return getattr(self, key)

	def __setitem__(self, key, data):
		return setattr(self, key, data)

	def __delitem__(self, key: str) -> None:
		self.__delattr__(key)

	def getNameOfVariable(self, variable):
		''' Returns the name of a variable
		:param variable: Value of the variable
		:returns: Name of the variable if found, else None.
		'''
		variableName = None
		for name in self.getVariables():
			if self[name] is variable:
				variableName = name
				break
		return variableName

	def check(self, allowEmptyVariables: bool = True):
		"""Check consistency of DataFrame

		Args:
			allowEmptyVariables (bool, optional): Allow variables to be empty, i.e. have size zero. Defaults to True.
		"""
		nEvents = None
		for variable in self.getLoadedNonscalarVariables():
			if self[variable].size == 0:
				continue
			if nEvents is None:
				nEvents = self[variable].shape[-1]
			elif self[variable].shape[-1] != nEvents:
				log.raiseException(Exception, f"Variable {variable} should have {nEvents}, but has {self[variable].shape[-1]} events!")
		for variable in self.getVariables():
			if not self.isScalar(variable):
				if not allowEmptyVariables and nEvents is not None and nEvents > 0 and (self[variable] is None or self[variable].size == 0):
					log.raiseException(Exception, f"Variable {variable} is empty, while there should be {nEvents} events!")
		return True

	def getLoadedVariables(self):
		'''Gets a list of variables that are currently loaded.
		'''
		return [variable for variable in self.getVariables() if getattr(self, variable) is not None]

	def getLoadedNonscalarVariables(self):
		'''Gets a list of non-scalar variables that are currently loaded.
		'''
		return [variable for variable in self.getVariables() if getattr(self, variable) is not None and not self.isScalar(variable, _checkHasVariable = False)]

	def getLoadedScalarVariables(self):
		'''Gets a list of scalar variables that are currently loaded.
		'''
		return [variable for variable in self.getVariables() if self.isScalar(variable, _checkHasVariable = False)]

	def getTotalNevents(self):
		'''Returns the number of total events in a sample

		:return: Number of total events
		'''
		variables = self.getLoadedNonscalarVariables()
		if variables:
			return self[variables[0]].shape[-1]
		return 0

	@property
	def nEvents(self):
		'''Property that returns the number of total events in a sample

		:return: Number of total events
		'''
		return self.getTotalNevents()

	def iadd(self, other: VariablesBase, moveOther: bool = False, check: bool = True) -> VariablesBase:
		"""Adds data from other Variables object to self.

		.. warning::
			This modifies self!

		Args:
			other (VariablesBase): Other variables object that will be added, i.e. concatenated to this one
			moveOther (bool, optional): Move the data of `other` to this object, i.e. DELETE the data stored in other. Defaults to False.
			                            This is a memory optimization for adding large variable objects and should be used with care
			check (bool, optional): Check self and other before adding them.

		Returns:
			VariablesBase: self
		"""
		if not isinstance(other, VariablesBase):
			log.raiseException(Exception, "Can iadd Variables instance only with another VariablesBase instance!")

		if check:
			self.check()
			other.check()

		# scalar variables
		thisLoadedVariables = set(self.getLoadedScalarVariables())
		otherLoadedVariables = set(other.getLoadedScalarVariables())
		commonLoadedVariables = thisLoadedVariables.intersection(otherLoadedVariables)
		variablesOnlyInOther = otherLoadedVariables.difference(thisLoadedVariables)
		for variable in commonLoadedVariables:
			if isinstance(self[variable], dict) and isinstance(other[variable], dict):
				for key, value in other[variable].items():
					if key not in self[variable]:
						self[variable][key] = value
					else:
						if not (np.isnan(self[variable][key]) and np.isnan(other[variable][key])) and self[variable][key] != other[variable][key]:
							log.raiseException(Exception, f"Dict entry {variable}[{key}] does not agree in both sets!")
			elif not (np.isnan(self[variable]) and np.isnan(other[variable])) and self[variable] != other[variable]:
				log.raiseException(Exception, f"Scalar variable {variable} does not agree in both sets")
		for variable in variablesOnlyInOther:
			self.addVariable(**other.getVariableInfo(variable))
			self[variable] = other[variable]

		# non-scalar variables
		thisLoadedVariables = set(self.getLoadedNonscalarVariables())
		otherLoadedVariables = set(other.getLoadedNonscalarVariables())
		commonLoadedVariables = thisLoadedVariables.intersection(otherLoadedVariables)
		variablesOnlyInThis = thisLoadedVariables.difference(otherLoadedVariables)
		variablesOnlyInOther = otherLoadedVariables.difference(thisLoadedVariables)
		indexOffset = self.getTotalNevents()
		for variable in commonLoadedVariables:
			self[variable] = np.hstack(( self[variable], other[variable] ))
			if moveOther:
				del other[variable]
		for variable in variablesOnlyInThis:
			shape = tuple( (iShape if i < (len(self[variable].shape)-1) else other.getTotalNevents()) for i, iShape in enumerate(self[variable].shape) )
			self[variable] = np.hstack(( self[variable], np.full( shape, np.nan, dtype=self[variable].dtype) ))
		for variable in variablesOnlyInOther:
			if variable not in self.getVariables():
				self.addVariable(**other.getVariableInfo(variable))
			shape = tuple( (iShape if i < (len(other[variable].shape)-1) else indexOffset) for i, iShape in enumerate(other[variable].shape) )
			self[variable] = np.hstack(( np.full( shape, np.nan, dtype=other[variable].dtype), other[variable] ))
			if moveOther:
				del other[variable]

		return self

	def __iadd__(self, other: VariablesBase):
		return self.add(other)

	def _obtainDataToStore(self) -> dict:
		"""Return data that should be stored in the `store` function.
		If data is stored to a pkl file, the whole object is pickled and this function is not used.
		"""
		return {v: self[v] for v in self.getVariables()}

	@classmethod
	def _createFromStoredData(cls, data: dict)->VariablesBase:
		"""Set data that was obtained using the `__obtainDataToStore` function
		"""
		variables = cls()
		for v, dataItem in data.items():
			VariablesBase.__setitem__(variables, v, dataItem)
		return variables

	def store(self, destinationPath: str, compressmethodHdf5: str = None, compresslevel: int = 1) -> None:
		"""Store the data on disk. Different data stores can be used depending on the file ending of `destinationPath`:

        - **.hdf5**: Stores in HDF5 format (default).
        - **.pkl**: Pickles the entire object and stores the binary dump.
        - **.gz**: GZips the pickled binary data.

        Args:
            destinationPath (str): Path to the output file.
            compressmethodHdf5 (str, optional): Compression method for HDF5 output files (None, 'lzf', 'gzip'). Defaults to no compression.
            compresslevel (int, optional): Compression level in case of gzip compression. Defaults to 1.
        """
		def storeDictToH5(fout: h5py.File, name: str, data: dict)-> None:
			dictGroup = fout.create_group(name)
			dictGroup.attrs['isPyDict'] = True
			for key, value in data.items():
				if value is None:
					log.warning(f"Cannot store data '{key}' as its value is None!. Ignoring it!")
					continue
				if not isinstance(key, str):
					log.raiseException(TypeError, "Store dictionary to h5 file allows only for string keys")
				dictGroup.create_dataset(key, data=value)

		if destinationPath.split('.')[-1] not in ['gz', 'pkl', 'hdf5', 'parquet']:
			destinationPath += '.hdf5' # use hdf5 per default

		if os.path.isfile(destinationPath):
			log.raiseException(Exception, f"{destinationPath} exists!")

		if destinationPath.endswith('.hdf5'):
			with h5py.File(destinationPath, 'w') as fout:
				data = self._obtainDataToStore()
				for key, item in data.items():
					if not isinstance(item, dict):
						if item is None:
							log.warning(f"Cannot store data '{key}' as its value is None!. Ignoring it!")
							continue
						fout.create_dataset(key, data=item, compression=compressmethodHdf5 if not np.isscalar(item) else None)
					else:
						storeDictToH5(fout, key, item)

		elif destinationPath.endswith('.parquet'):
			dataFrame = self.toPolarsDataFrame()
			dataFrame.write_parquet(destinationPath, compression='zstd')

		else: # pickle or gzip pickle
			if destinationPath.endswith('.gz') and compresslevel > 0:
				openFunction = lambda filepath: gzip.open(filepath, 'wb', compresslevel=compresslevel)
			else:
				openFunction = lambda filepath: open(filepath, 'wb')   # pylint: disable=consider-using-with
			destinationPath = os.path.abspath(destinationPath)
			denstinationDir = os.path.dirname(destinationPath)
			if not os.path.exists(denstinationDir):
				os.makedirs(denstinationDir)

			with openFunction(destinationPath) as fout:
				pickle.dump(self, fout, protocol=4)

	@classmethod
	def load(cls, sourcePath: str, printInfo: bool =True, excludeVariables: list=None, onlyIncludeVariables: list=None) -> VariablesBase:
		"""Loads a saved variables object.

		Args:
			sourcePath (str): Path to file, in which the variables object is saved.
			printInfo (bool, optional): Prints the amount of events loaded from the files. Defaults to True.
			excludeVariables (list, optional): List of variable names that shall not be loaded. Defaults to loading all variables.
			onlyIncludeVariables (list, optional): If a list is given, only the variables mentioned within the list are loaded. Defaults to loading all variables.

		Returns:
			VariablesBase
		"""
		def loadDatasetFromH5(dataset: h5py.File):
			data = None
			if 'isPyDict' in dataset.attrs and dataset.attrs['isPyDict']: # this is a group which represents a dict
				data = {}
				for key in dataset:
					data[key] = loadDatasetFromH5(dataset[key])
			elif dataset.ndim > 0:
				data = dataset[:].copy()
			else:
				data = copy.deepcopy(dataset[()])
				if isinstance(data, bytes):
					data = data.decode()
			return data

		variables = None
		if sourcePath.endswith('.hdf5'):
			with h5py.File(sourcePath, 'r') as fin:
				data = {}
				if onlyIncludeVariables is not None:
					onlyIncludeVariables = set(onlyIncludeVariables)
					if cls is Variables:
						onlyIncludeVariables = onlyIncludeVariables.union(['__4MomentumVars','__branchnames','__auto',
																		'__castDtype','__channelLabels','__channelMaps',
																		'__channelOrder','__is4Momentum'])
					for key in onlyIncludeVariables:
						try:
							data[key] = loadDatasetFromH5(fin[key])
						except KeyError:
							log.raiseException(KeyError, f"Variable '{key}' not found in file '{sourcePath}'!")
				else:
					for key in fin:
						if excludeVariables is not None and key in excludeVariables:
							continue
						data[key] = loadDatasetFromH5(fin[key])
				variables = cls._createFromStoredData(data)
		elif sourcePath.endswith('.gz') or sourcePath.endswith('.pkl'):
			if sourcePath.endswith('.gz'):
				openFunction = lambda filepath: gzip.open(filepath, 'rb')
			else:
				openFunction = lambda filepath: open(filepath, 'rb')   # pylint: disable=consider-using-with
			with openFunction(sourcePath) as fin:
				variables = pickle.load(fin)
		elif sourcePath.endswith('.parquet'):
			columns = onlyIncludeVariables
			if columns is None and excludeVariables is not None:
				excludedColumns = set(excludeVariables)
				for variable in excludeVariables:
					excludedColumns.update(f'{variable}_{component}' for component in MOMENTUM_COMPONENTS)
				columns = [
					columnName for columnName in pl.read_parquet_schema(sourcePath)
					if columnName not in excludedColumns
				]
			dataFrame = pl.read_parquet(sourcePath, columns=columns, use_pyarrow=True)
			variables = cls.fromPolarsDataFrame(dataFrame)
		if printInfo:
			log.info(f"Loaded data {variables.getTotalNevents():,} events from '{sourcePath}'")
		return variables

	def filter(self, filterMask, copy=True) -> VariablesBase:
		'''Filter the data with the option to create a copy or modify the calling object.

		:param filterMask: This can be an array of booleans of size nEvents or a list of indices in the range [0, nEvents).
		:param copy: If copy is set to False, the calling object is modified instead of copying all data to a new object before filtering. Defaults to True.
		'''
		if callable(filterMask):
			return self.filter(filterMask(self), copy)
		return self._filterWithMask(filterMask, copy)

	def _filterWithMask(self, filterMask, copy) -> VariablesBase:
		nEvents = self.nEvents
		if copy:
			variables = self.copy(doNotCopyNonscalarVariables=True)
		else:
			variables = self

		for variable in self.getLoadedNonscalarVariables():
			if self[variable].size > 0:
				if issubclass(filterMask.dtype.type, np.integer):
					if filterMask.min() >= 0 and filterMask.max() < nEvents:
						variables[variable] = self[variable][...,filterMask].copy()
					else:
						log.raiseException(Exception, f'Filter mask of indices out of range [{filterMask.min()}, {filterMask.max()}]')
				else:
					if self[variable].shape[-1] == filterMask.shape[-1]:
						variables[variable] = self[variable][...,filterMask].copy()
					else:
						log.raiseException(Exception, f'Cannot filter variable {variable}, because variable has shape {variables[variable].shape}, but filter map has shape {filterMask.shape}')
			else:
				variables[variable] = self[variable].copy()
		return variables

	def splitRandomYield(self, subsetFractions: Sequence[float], shuffle: bool = False) -> collections.abc.Iterator[VariablesBase]:
		"""Split this data set into (len(subsetFractions)+1) sub-sets with given fractions by randomly selecting events.

		The default generator from `sphysics.random.generators` is used.

		Args:
			subsetFractions (Sequence[float]): Fraction for all subset, except for the last one. The last subset contains the rest of the evetns
			shuffle (Sequence[float], optional): Shuffle events. If false the order of events in each random subset is preserved.

		Returns:
			Iterator[VariablesBase]: Iterator of subsets, which are randomly selected copies of this data set
		"""
		if np.sum(subsetFractions) >= 1.0:
			log.raiseException(Exception, "Sum of subsetFractiosn is {0}, i.e. larger than or equal one!", np.sum(subsetFractions))

		subsetsNevents = [ round(self.nEvents*subsetFraction) for subsetFraction in subsetFractions ]
		subsetRanges = [0]
		for subsetNevetns in subsetsNevents:
			subsetRanges.append(subsetRanges[-1] + subsetNevetns)
		subsetRanges.append(self.nEvents)
		nSubsets = len(subsetRanges)-1

		shuffledIndices = np.arange(self.nEvents)
		random.generators.default.shuffle(shuffledIndices)

		for iSubset in range(nSubsets):
			subset = shuffledIndices[subsetRanges[iSubset]:subsetRanges[iSubset+1]]
			if not shuffle:
				filterMask = np.zeros(self.nEvents, dtype=bool)
				filterMask[subset] = True
			else:
				filterMask = subset
			subset = self.filter(filterMask, copy=True)
			yield subset

	def splitRandom(self, subsetFractions: Sequence[float], shuffle: bool = False) -> list[VariablesBase]:
		"""Split this dataset into (len(subsetFractions)+1) subsets with given fractions by randomly selecting events.

		The order of the events in each random subset is preserved, i.e. the events are not shuffled.
		The default generator from `sphysics.random.generators` is used.

		Args:
			subsetFractions (Sequence[float]): Fraction for all subset, except for the last one. The last subset contains the rest of the evetns
			shuffle (Sequence[float], optional): Shuffle events. If false the order of events in each random subset is preserved.

		Returns:
			list[VariablesBase]: List of subsets, which are randomly selected copies of this data set
		"""
		return list(self.splitRandomYield(subsetFractions, shuffle=shuffle))

	def toDataFrame(self) -> pd.DataFrame:
		"""Convert the Variables object to a Pandas DataFrame

		If a variable is 2-dimensional, i.e. :math:`(<i>, nEvents)`, then a
		column for each :math:`<i>` is created in the DataFrame and the columns are
		named :math:`<var>[<i>]`.

		If a variable is `None` or has dimension > 2, it will not be
		added to the data frame.

		Returns:
			pd.DataFrame: DataFrame with all the variables
		"""
		columns = [ var for var in self.getVariables() if self[var] is not None and self[var].ndim==1]
		data = [ self[var] for var in columns]
		for var in self.getVariables():
			if var in columns:
				continue
			if self[var] is not None and self[var].ndim==2:
				for i in range(self[var].shape[0]):
					columns.append(f'{var}[{i}]')
					data.append(self[var][i])
			else:
				log.warning(f"Cannot convert variable '{var}' to Pandas Dataframe")
		data = np.vstack(data).T
		df = pd.DataFrame(data, columns=columns)
		return df

	@classmethod
	def fromDataFrame(cls, df: pd.DataFrame, copy: bool = True, dropColumns: bool = False) -> VariablesBase:
		"""Convert data frame to variables object

		Args:
			df (pd.DataFrame): Data frame to be converted to variables object
			copy (bool, optional): Copy the data even if it is not strictly necessary.
			dropColumns (bool, optional): Drop columns from input data frame as soon as they are added to the Variables object.
			                              CHANGES THE INPUT DATAFRAME. Can be very slow. Defaults to False.

		Returns:
			VariablesBase: Variables object with data
		"""
		variables = cls()
		for variable in df.columns:
			variables.addVariable(str(variable))
			variables[str(variable)] = df[variable].to_numpy(copy=copy)
			if dropColumns:
				df.drop(variable)
		return variables

	def toPolarsDataFrame(self) -> pl.DataFrame:
		"""Convert this object to a Polars DataFrame and release its array data.

		One-dimensional arrays retain their variable names. Ordinary two-dimensional
		arrays are stored as ``<name>[<index>]`` columns. The Variables object is
		consumed after successful conversion and must not be reused for event data.
		"""
		nEvents = self.nEvents
		series = []
		for variable in self.getVariables():
			values = self[variable]
			if values is None:
				log.warning(f"Cannot convert variable '{variable}' to Polars Dataframe")
				continue
			variableSeries = self._toPolarsSeries(variable, values, nEvents)
			if variableSeries is not None:
				series.extend(variableSeries)
				self[variable] = None

		return pl.DataFrame(series)

	def _toPolarsSeries(self, variable: str, values, nEvents: int) -> list[pl.Series] | None:
		"""Create Polars series for one non-special variable without copying its data."""
		if self.isScalar(variable):
			if isinstance(values, dict):
				log.warning(f"Cannot convert variable '{variable}' to Polars Dataframe")
				return None
			return [pl.Series(variable, np.full(nEvents, values))]
		if values.ndim == 1:
			return [pl.Series(variable, values)]
		if values.ndim == 2:
			return [
				pl.Series(f'{variable}[{index}]', row)
				for index, row in enumerate(self._rowViews(values))
			]
		log.warning(f"Cannot convert variable '{variable}' to Polars Dataframe")
		return None

	@staticmethod
	def _rowViews(values: np.ndarray) -> list[np.ndarray]:
		"""Return contiguous row views that Polars can adopt without copying."""
		if not values.flags.c_contiguous:
			values = np.ascontiguousarray(values)
		return list(values)

	@classmethod
	def fromPolarsDataFrame(cls, dataFrame: pl.DataFrame, zeroCopy: bool = True) -> VariablesBase:
		"""Create Variables from a Polars DataFrame, consuming its columns.

		One-dimensional variables can share Arrow buffers when ``zeroCopy`` is
		true; buffers that cannot be exposed as NumPy arrays are copied. Subclasses
		can consume special column groups before ordinary columns are transferred.
		"""
		variables = cls()
		variables._fromPolarsSpecialColumns(dataFrame)
		for columnName in list(dataFrame.columns):
			variables.addVariable(columnName)
			variables[columnName] = _moveColumnToNumpy(dataFrame, columnName, zeroCopy)
		return variables

	def _fromPolarsSpecialColumns(self, dataFrame: pl.DataFrame) -> None:
		"""Consume subclass-specific Polars column groups before ordinary columns."""

	@classmethod
	def concat(cls, variablesSequence: Sequence[VariablesBase]) -> VariablesBase:
		"""Concatenates the given sequence of variables.

		The set of loaded non-scalar variables need to be the same in each Variables object in the set.
		If this is not the case, use the += operator, which works in all cases

		.. warning::
			The input variables objects in the sequence WILL BE DELETED!

		Args:
			variablesSequence (Sequence[VariablesBase]): Sequence of variables to be concatenated

		Returns:
			VariablesBase: Concatenated Variables object
		"""
		variableNames = variablesSequence[0].getLoadedNonscalarVariables()[1:] # first variable needed to remember number of events
		for variablesI in variablesSequence[1:]:
			if variablesI.getLoadedNonscalarVariables()[1:] != variableNames:
				raise Exception("Works only of all variable objects have the same set of loaded non-scalar variables")

		data = {} # temporary data store for non-scalar variables
		for variableName in variableNames:
			data[variableName] = []
			for variablesI in variablesSequence:
				data[variableName].append(variablesI[variableName])
				variablesI[variableName] = None
		# merge variables objects
		variables = variablesSequence[0]
		for variablesI in variablesSequence[1:]:
			variables.iadd(variablesI, moveOther=True, check=False)

		# concatenate and set data
		for variableName in variableNames:
			shape = data[variableName][0].shape
			for subdata in data[variableName][1:]:
				if subdata.shape[:-1] != shape[:-1]:
					log.raiseException(Exception, f'Data of variable {variableName} have different shapes {shape} != {subdata.shape}. Cannot concat samples.')
			variables[variableName] = np.hstack(data[variableName])
			for i in reversed(range(len(data[variableName]))):
				del data[variableName][i]
			del data[variableName]
		variablesSequence.clear()
		return variables


class Variables(VariablesBase):
	'''
	Extends :class:`VariablesBase` by channels, loading from ROOT Trees and more.
	'''
	def __init__(self) -> None:
		super().__init__()
		self.__branchnames = {}
		self.__auto = {}
		self.__channelMaps = {}
		self.__is4Momentum = {}
		self.__4MomentumVars = {}
		self.__castDtype = {}
		self.__channelOrder = []
		self.__channelLabels = {}

	def __setitem__(self, key, data):
		'''Sets a variable

		:param key: Name of the variable
		:param data: Data to be assigned to the variable
		'''
		if not self.hasVariable(key):
			raise Exception(f"Variable '{key}' not stored")
		return super().__setitem__(key, data)

	def getBranchname(self, channel):
		'''Retrieves the name of the Branch for a given channel.

		:param channel: The name of the channel
		'''
		return self.__branchnames[channel]

	def getAuto(self, channel):
		'''Checks if the variable is loaded automatically

		:param channel: The name of the channel
		:return: True, if the variable is automatically loaded, else False
		'''
		return self.__auto[channel]

	def getIs4Momentum(self, name):
		'''Checks if the variable represents a four-momentum stored in 4 individual branches.

		:param name: Name of the variable
		:return: True, if the variable is a four-momentum, else False
		'''
		return self.__is4Momentum[name] if name in self.__is4Momentum else False

	def get4MomentumVars(self, name):
		'''Gets the name of the four-momentum variables

		:param name: Name of the variable
		:return: List of four momentum variable names
		'''
		return self.__4MomentumVars[name]

	def getVariableInfo(self, name) -> dict:
		'''Gets all the variable information for `addVariable`

		:param name: Name of the variable
		:return: Dictionary containing variable information
		'''
		kwargs = super().getVariableInfo(name)
		kwargs.update({
		    "branchname": self.__branchnames[name],
		    "auto": self.__auto[name],
		    "is4Momentum": self.__is4Momentum[name],
		    'momentumVariables': self.__4MomentumVars[name],
		    "castDtype": self.__castDtype[name]
		})
		return kwargs

	def addVariable(self, name, branchname = None, auto=True, is4Momentum=False, momentumVariables = None, castDtype = False) -> bool: # pylint: disable=arguments-differ
		'''Adds a new variable

		:param name:  Name of the variable
		:param branchname: branch name (equals name if not given)
		:param auto: Whether it is automatically loaded from the tree, defaults to True
		:param is4Momentum: Whether it is a four-momentum variable, defaults to False
		:param momentumVariables: Names of the 4 variables of the four-momentum, Defaults to :math:`[E, p_x, p_y, p_z]`
		:param castDtype: Cast array to a given type when loading form a root file, defaults to False.
		Special types \\: boolNanfalse \\:Same as bool, but casts `nan` to False instead of True
		:return: True, if the variable is added correctly, else False
		'''
		if super().addVariable(name):
			self.__branchnames[name] = branchname if branchname is not None else name
			self.__auto[name] = auto
			self.__is4Momentum[name] = is4Momentum
			self.__castDtype[name] = castDtype
			if momentumVariables is not None:
				self.__4MomentumVars[name] = momentumVariables
			else:
				self.__4MomentumVars[name] = ['E', 'px', 'py', 'pz']
			return True
		return False

	def _toPolarsSeries(self, variable: str, values, nEvents: int) -> list[pl.Series] | None:
		"""Create Polars series for one variable, including four-momenta."""
		if not self.getIs4Momentum(variable):
			return super()._toPolarsSeries(variable, values, nEvents)
		if values.ndim != 2 or values.shape[0] != len(MOMENTUM_COMPONENTS):
			log.raiseException(ValueError, f"Four-momentum variable '{variable}' must have shape (4, nEvents)")
		return [
			pl.Series(f'{variable}_{component}', row)
			for component, row in zip(MOMENTUM_COMPONENTS, self._rowViews(values))
		]

	def _fromPolarsSpecialColumns(self, dataFrame: pl.DataFrame) -> None:
		"""Consume complete four-momentum column quartets from ``dataFrame``."""
		columnNames = set(dataFrame.columns)
		momentumNames = sorted({
			columnName.removesuffix(f'_{MOMENTUM_COMPONENTS[0]}') for columnName in columnNames if columnName.endswith(f'_{MOMENTUM_COMPONENTS[0]}')
			and all(f'{columnName.removesuffix(f"_{MOMENTUM_COMPONENTS[0]}")}_{component}' in columnNames for component in MOMENTUM_COMPONENTS)
		})
		for name in momentumNames:
			componentNames = [f'{name}_{component}' for component in MOMENTUM_COMPONENTS]
			emptyFrame = dataFrame.head(0) # cheap way to get the exact numpy dtype of each column
			dtype = np.result_type(*(emptyFrame[componentName].to_numpy().dtype for componentName in componentNames))
			momentum = np.empty((len(MOMENTUM_COMPONENTS), dataFrame.height), dtype=dtype)
			for index, componentName in enumerate(componentNames):
				values = _moveColumnToNumpy(dataFrame, componentName, zeroCopy=True)
				momentum[index] = values
				del values # release the component buffer before the next one is moved out
			self.addVariable(name, is4Momentum=True)
			self[name] = momentum

	def insertTree(self, channel, tree, entryStart: int | None = None, entryStop: int | None = None):
		'''Insert the given ROOT tree into this object, adding all branches of the tree as variables.

		:param channel: Channel to which the data in the three should be assigned
		:param tree: ROOT tree to be added to this object
		:param entryStart: First ROOT entry to load, inclusive. Defaults to the first entry.
		:param entryStop: First ROOT entry not to load. Defaults to the entry after the last one.
		'''
		variables = [ var for var, isAuto in self.__auto.items() if isAuto
		                                                         and self.__branchnames[var] in tree.keys()
		               and not self.__is4Momentum[var]]
		branchNames = [ self.__branchnames[var] for var in variables ]
		for branchName in branchNames:
			if branchName not in tree.keys():
				log.raiseException(Exception, f"Branch {branchName} is not in the tree!")
		data = tree.arrays(branchNames, library='np', entry_start=entryStart, entry_stop=entryStop,
		                   decompression_executor=uproot.ThreadPoolExecutor(NUM_UPROOT_THREADS))
		for variable, branchname in zip(variables, branchNames):
			self.insertData(variable, channel, data[branchname])

		variables = [ var for var, isAuto in self.__auto.items() if isAuto
		               and self.__is4Momentum[var]]
		for var in variables:
			self.insert4momentum(var, channel, tree, entryStart=entryStart, entryStop=entryStop)

	def insert4momentum(self, name: str, channel: str, tree, entryStart: int | None = None, entryStop: int | None = None):
		'''Insert the four momentum variables stored in the given tree with the name `name`.

		:param name: Name of the momentum variable
		:param channel: Channel to which the data should be added
		:param tree: The ROOT tree from which the momentum variables will be added to this object
		:return: True, if the four-momentum was inserted correctly, False otherwise
		'''
		postfix = ''
		if name.endswith('_CMS'):
			postfix = '_CMS'
		basename = '_'.join(self.__branchnames[name].rstrip('_CMS').split('_')[:-1])
		momentumVariables = self.__4MomentumVars[name]
		branchnames = [ f'{basename}_{v}{postfix}' for v in momentumVariables ]
		for branchName in branchnames:
			if branchName not in tree.keys():
				log.debug(f"Branch {branchName} is not in the tree, cannot load '{name}'")
				return False
		data = tree.arrays(branchnames, library='np', entry_start=entryStart, entry_stop=entryStop,
		                   decompression_executor=uproot.ThreadPoolExecutor(NUM_UPROOT_THREADS))
		p = np.vstack([ data[f"{basename}_{v}{postfix}"] for v in momentumVariables ])
		self.insertData(name, channel, p)
		return True

	def insert(self, name, channel, tree):
		'''Insert variable with name `name` from the given ROOT tree.

		:param name: Name of the variable
		:param channel: Channel to which the loaded data should be added
		:param tree: ROOT tree with the variable
		'''
		if self.__branchnames[name] in tree.keys():
			data = tree.arrays(self.__branchnames[name], library='np', decompression_executor=uproot.ThreadPoolExecutor(NUM_UPROOT_THREADS))[self.__branchnames[name]]
			self.insertData(name, channel, data)

	def insertData(self, name, channel, data):
		'''Insert variable with name `name` with the given data.

		:param name: Name of the variable
		:param channel: Channel to which the loaded data should be added
		:param data: Data of the variable
		:return: True if variable was added successfully, False otherwise
		'''
		if self[name] is not None:
			log.raiseException(Exception, f"Data for {name} already inserted")
		indices = np.arange(data.shape[-1], dtype=int)

		if channel in self.__channelMaps and \
                           (self.__channelMaps[channel].size != indices.size or np.intersect1d(self.__channelMaps[channel], indices).size != indices.size):
			log.raiseException(Exception, f"Index map for variable {name} in channel {channel} does not match previously added data for this index")

		if name in self.__castDtype and self.__castDtype[name]:
			if self.__castDtype[name] == 'boolNanfalse':
				self[name] = (data > 0) | (data < 0)
			else:
				with np.errstate(all='raise'):
					try:
						self[name] = data.astype(self.__castDtype[name], copy=True)
					except FloatingPointError as e:
						log.warning(f"Casting '{name}' to '{str(self.__castDtype[name])}: " + str(e))
						log.warning(str(data))
						raise e
		else:
			self[name] = np.copy(data)
		if channel not in self.__channelMaps:
			self.__channelMaps[channel] = indices
		return True

	def getNevents(self):
		''' Gets the number of events per channel.

		:return: {<channel>: <events-in-channel>}. If the number of events is different for different variables, <events-in-channel> is False
		'''
		nEvents = {}
		for channel in self.getChannels():
			nEvents[channel] = self.__channelMaps[channel].shape[-1]
		return nEvents

	def getWeightedNevents(self, weight: str|np.ndarray= None) -> dict:
		'''Calculate the number of events per channel taking into account the weights

		:param weight: Name of weight member variable or array of weights. Defaults to 'weight'.
		:return: Array with channel name as key and number of events as value'''
		if weight is None:
			weight = 'weight'
		if not isinstance(weight, np.ndarray):
			if not self.hasVariable(weight):
				log.raiseException(KeyError, f'Sample does not have weight variable "{weight}"!')
			weight = self[weight]
		return {channel: math.sum(weight[self.idx(channel)]) for channel in self.getChannels()}

	def emptyCopy(self) -> Variables:
		'''Creates an empty copy of the Variables object

		:return: A new Variables object with the same structure but no data. '''
		newVariables = Variables()
		for variable in self.getVariables():
			newVariables.addVariable(**self.getVariableInfo(variable))
		return newVariables

	def idx(self, channel):
		'''Gets the index map for a given channel

		:param channel: The channel name
		:return: The index map '''
		return self.__channelMaps[channel]

	def setChannelIndexMap(self, channel, indexMap):
		'''Sets the index map for a given channel

		:param channel: The channel name
		:param indexMap: Index map associated with the channel '''
		self.__channelMaps[channel] = indexMap

	def delChannelIndexMap(self, channel):
		'''Deletes the index map for a given channel'''
		del self.__channelMaps[channel]

	def getChannels(self):
		'''Retrieve a list of channels

		:return: Sorted list of channel names '''
		if self.__channelOrder is None or len(self.__channelOrder) == 0:
			return sorted(list(self.__channelMaps.keys()))
		return sorted([ c for c in self.__channelMaps if c in self.__channelOrder ],
		key = self.__channelOrder.index) \
        + sorted([ c for c in self.__channelMaps if c not in self.__channelOrder ])

	def mergeChannels(self, channelGroups: dict[str, list[str]]):
		'''Merges multiple channels into one

		:param channelGroups: Dictionary with group names and list of channels that should be merged into this group
		:type channelGroups: dict[str, list[str]]
		:return: An updated Variables object with merged channels
		'''
		for channelGroupName, channelGroup in channelGroups.items():
			for channel in channelGroup:
				if channel not in self.getChannels():
					log.raiseException(KeyError, f"Channel '{channel}' of group '{channelGroupName}' not in Variables object!")
			idx = functools.reduce(np.union1d, (self.idx(c) for c in channelGroup))
			self.setChannelIndexMap(channelGroupName, idx)
			for channel in channelGroup:
				if channel not in channelGroups.keys():
					self.delChannelIndexMap(channel)
		return self

	def setChannelOrder(self, channelOrder: list) -> None:
		"""Set the ordering of channels

		Channels are ordered in the same orders as in `channelOrder`.
		Channels that are not included in `channelOrder` are orders alphabetically.

		:param channelOrder: List of channels names.
		"""
		self.__channelOrder = channelOrder

	def setChannelLabels(self, channelLabels: dict[str,str]) -> None:
		"""Set labels for each channel that can be used for plotting, ....

		:param channelLabels (dict[str,str]): key is the channel name, value is the label
		"""
		self.__channelLabels = channelLabels

	def getChannelLabel(self, channelName: str) -> str:
		"""Get label for given channel name. Return channel name if no label is set.
		"""
		return self.__channelLabels.get(channelName, channelName)


	def check(self, allowEmptyVariables: bool = True):
		"""Check consistency of data frame

		Args:
			allowEmptyVariables (bool, optional): Allow variables to be empty, i.e. have size zero. Defaults to False.
		"""
		super().check(allowEmptyVariables=allowEmptyVariables)
		for channel in self.getChannels():
			if self.getNevents()[channel] != np.unique(self.__channelMaps[channel]).size:
				log.raiseException(Exception, f"Indices in mapping for channel {channel} appear multiple times!")
			if self.__channelMaps[channel].size > 0 and np.max(self.__channelMaps[channel]) >= self.getTotalNevents():
				log.raiseException(Exception, f"Largest index in mapping for channel {channel} larger than number of events!")
		return True

	def _obtainDataToStore(self) -> dict:
		data = super()._obtainDataToStore()
		privateVars = ['__branchnames', '__auto', '__channelMaps', '__is4Momentum', '__4MomentumVars', '__castDtype', '__channelOrder', '__channelLabels']
		for var in privateVars:
			data[var] = getattr(self, f'_Variables{var}')
		return data

	@classmethod
	def _createFromStoredData(cls, data: dict) -> VariablesBase:
		privateVars = ['__branchnames', '__auto', '__channelMaps', '__is4Momentum', '__4MomentumVars', '__castDtype', '__channelOrder', '__channelLabels']
		privateData = {}
		for var in privateVars:
			if var not in data:
				if var == '__channelOrder':
					privateData[var] = []
				elif var == '__channelLabels':
					privateData[var] = {}
				else:
					raise Exception(f"Variable '{var}' not found in loaded data!")
			else:
				if var == '__channelOrder':
					privateData[var] = list(data[var])
				else:
					privateData[var] = data[var]
				del data[var]
		variables = super()._createFromStoredData(data)
		for var, varData in privateData.items():
			if var == '__4MomentumVars':
				varData = {name: [branch.decode() for branch in branches] for name, branches in varData.items()}
			setattr(variables, f'_Variables{var}', varData)
		return variables

	def iadd(self, other: Variables, moveOther: bool = False, check: bool = True) -> VariablesBase:
		"""Adds data from other Variables object to self.

		.. warning::
			This modifies self!

		Args:
			other (VariablesBase): Other variables object that will be added, i.e. concatenated to this one
			moveOther (bool, optional): Move the data of `other` to this object, i.e. DELETE the data stored in other. Defaults to False.
			                            This is a memory optimization for adding large variable objects and should be used with care
			check (bool, optional): Check self and other before adding them.

		Returns:
			VariablesBase: self
		"""
		if not isinstance(other, Variables):
			log.raiseException(Exception, "Can iadd Variables instance only with another Variables instance!")
		indexOffset = self.getTotalNevents()

		super().iadd(other, moveOther=moveOther, check=check)

		for channel in other.getChannels():
			if channel in self.__channelMaps:
				self.__channelMaps[channel] = np.hstack([self.__channelMaps[channel], other.idx(channel) + indexOffset])
			else:
				self.__channelMaps[channel] = other.idx(channel) + indexOffset

		return self

	def __iadd__(self, other: Variables):
		return self.iadd(other)

	def printSummary(self):
		'''Prints a summary of the variables and channels
		'''
		log.info("Variables:")
		log.incrementIndent()
		for variable in self.getVariables():
			log.info(variable)
		log.decrementIndent()
		nEvents = self.getNevents()
		log.info("Channels:")
		log.incrementIndent()
		for channel in sorted(self.getChannels()):
			log.info('{0:10s}{1:>13,d}'.format(channel, nEvents[channel]))
		log.decrementIndent()

	def _filterWithMask(self, filterMask, copy=True) -> Variables:
		''' If copy is set to False, the calling object is modified instead of copying all data to a new object before filtering
		'''
		nEvents = self.nEvents
		variables = super()._filterWithMask(filterMask, copy)

		idxMap = np.full(nEvents, -1, dtype=int)
		if issubclass(filterMask.dtype.type, np.integer):
			idxMap[filterMask] = np.arange(filterMask.size, dtype=int)
		else:
			idxMap[filterMask] = np.arange(np.sum(filterMask), dtype=int)

		for channel in variables.getChannels():
			idxMapOfChannel = idxMap[variables.idx(channel)]
			nSelectedOfChannel = np.sum(idxMapOfChannel!=-1)
			variables.setChannelIndexMap(channel, idxMapOfChannel[idxMapOfChannel>=0])
			if variables.idx(channel).size != nSelectedOfChannel:
				raise Exception("Wrong size of index map")
		return variables

	def loadFromRootFiles(self,
					   inputFilePaths: Sequence[str],
					   treeName: str, channel: str,
					   fast: bool = False,
					   quiet: bool = False,
					   allBranches: bool = False,
					   entryStart: int | None = None,
					   entryStop: int | None = None) -> Variables:
		"""Load variables from ROOT trees in input files

		Args:
			inputFilePaths (Sequence[str]): List of paths to input files
			treeName (str): Name/Path of the tree in the input file
			channel (str): Name of the channel to which this data is added
			fast (bool): Fast merging of loaded ROOT files using `Variables.concat`,
			            assuming that the trees in all root files are the same, i.e. contain the same branches. This option has to be used with care.
			quiet (bool): No printout
			allBranches (bool): Load all branches as listed in the ROOT tree in the first file
			entryStart (int, optional): First ROOT entry to load from each file, inclusive.
			entryStop (int, optional): First ROOT entry not to load from each file.
		"""
		if not quiet:
			log.initProgress(len(inputFilePaths))
		loadedVariables = []
		for iInputFile, inputFilePath in enumerate(inputFilePaths):
			with uproot.open(inputFilePath) as inputFile:
				if iInputFile==0 and allBranches:
					tree = inputFile[treeName]
					for branch in tree.keys():
						if not self.hasVariable(branch):
							self.addVariable(branch)
				selfCopy = self.emptyCopy()
				selfCopy.insertTree(channel, inputFile[treeName], entryStart=entryStart, entryStop=entryStop)
				if not fast:
					self.iadd(selfCopy, moveOther=True)
				else:
					loadedVariables.append(selfCopy)
			if not quiet:
				log.updateProgress()
		if not quiet:
			log.finishProgress()
		if fast:
			others = Variables.concat(loadedVariables)
			self.iadd(others, moveOther=True)
			del others
		return self
