# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Utils for PWA fits
'''

import copy
import yaml
import numpy as np

from ...utils import Logger

log = Logger('PWA')


class BinningCell(object):
	"""Class to hold a multi-dimensional Cell."""

	def __init__(self, boundaries):
		"""
		Initialize a BinningCell.

		Parameters
		----------
		boundaries : dict or BinningCell
			Dictionary of binning variable names to (lower, upper) tuples,
			or another BinningCell to copy.
		"""
		if isinstance(boundaries, BinningCell):
			boundaries = copy.copy(boundaries.boundaries)
		elif isinstance(boundaries, dict):
			boundaries = {k:tuple(v) for k, v in boundaries.items()}
		else:
			raise TypeError("Boundaries is not of type 'dict'.")

		for key in boundaries.keys():
			if not isinstance(key, str):
				raise TypeError("Binning variable name is not a string type.")
			binRange = boundaries[key]
			if not isinstance(binRange, tuple):
				raise TypeError("Binning range is not of type 'tuple' for binning variable '" + key + "'.")
			if len(binRange) != 2:
				raise ValueError("Binning range does not have two entries for binning variable '" + key + "'.")
			if not isinstance(binRange[0], (float,int)):
				raise TypeError("Lower bound of bin range is not a number for binning variable '" + key + "'.")
			if not isinstance(binRange[1], (float,int)):
				raise TypeError("Upper bound of bin range is not a number for binning variable '" + key + "'.")
			if binRange[0] > binRange[1]:
				raise ValueError("Lower bound of bin range (" + str(binRange[0]) + ") is larger than upper bound of bin range (" + str(binRange[1]) + ").")
			if isinstance(binRange[0], int) or isinstance(binRange[1], int):
				boundaries[key] = (float(binRange[0]), float(binRange[1]))
		self.boundaries = boundaries

	def __lt__(self, other):
		"""
		Compare if this cell center is lower than another cell center

		Parameters
		----------
		other : BinningCell
			The cell to compare with.

		Returns
		-------
		bool
			True if this cell is lower than the other, False otherwise.
		"""
		keys = sorted(self.boundaries.keys())
		if keys != sorted(other.boundaries.keys()):
			return False
		for key in keys:
			selfCenter = (self.boundaries[key][0] + self.boundaries[key][1]) / 2.
			otherCenter = (other.boundaries[key][0] + other.boundaries[key][1]) / 2.
			if selfCenter > otherCenter:
				return False
			if selfCenter < otherCenter:
				return True
		return False

	def __le__(self, other):
		"""
		Compare if this cell center is lower than or equal another cell center

		Parameters
		----------
		other : BinningCell

		Returns
		-------
		bool
		"""
		return self < other or self == other


	def __eq__(self, other):
		"""
		Check if this cell is equal to another cell.

		Parameters
		----------
		other : BinningCell

		Returns
		-------
		bool
		"""
		return self.boundaries == other.boundaries


	def __ne__(self, other):
		"""
		Check if this cell is not equal to another cell.

		Parameters
		----------
		other : BinningCell

		Returns
		-------
		bool
		"""
		return not self == other


	def __gt__(self, other):
		"""
		Compare if this cell is greater than another cell.

		Parameters
		----------
		other : BinningCell

		Returns
		-------
		bool
		"""
		return not self <= other


	def __ge__(self, other):
		"""
		Compare if this cell is greater than or equal to another cell.

		Parameters
		----------
		other : BinningCell

		Returns
		-------
		bool
		"""
		return not self < other


	def __str__(self):
		retval = "Cell: " + self.__repr__()
		return retval

	def __repr__(self):
		retVariables = []
		for key in sorted(self.boundaries.keys()):
			retVariables.append( "\"" + key + "\": (" + str(self.boundaries[key][0]) + ", " + str(self.boundaries[key][1]) + ")")
		retval = "{ "
		retval += ", ".join(retVariables)
		retval += " }"
		return retval

	def __hash__(self):
		keys = tuple(self.variables())
		values = tuple(self.boundaries[k] for k in keys)
		return hash((keys, values))

	def __contains__(self, other):
		"""
		Check if another cell is contained in this cell.

		The other Cell is contained in this Cell if the intervals of all variables
		of this Cell are larger than the corresponding intervals of the other Cell.
		If this Cell is binned in a variable in which the other Cell is not binned,
		the other Cell is not contained in this Cell.
		If the other Cell is binned in a variable in which this Cell is not binned,
		the other Cell is contained in this Cell.

		Parameters
		----------
		other : BinningCell or dict

		Returns
		-------
		bool
		"""
		if isinstance(other, dict):
			other = BinningCell(other)
		if not isinstance(other, BinningCell):
			msg = "Cannot check whether object of type {0} is contained in a Cell".format(type(other))
			log.error(msg)
			raise ValueError(msg)

		for key in list(self.boundaries.keys()):
			if key in list(other.boundaries.keys()):
				if not self.boundaries[key][0] <= other.boundaries[key][0]:
					return False
				if not self.boundaries[key][1] >= other.boundaries[key][1]:
					return False
			else:
				return False
		return True

	def variables(self):
		"""
		Get the sorted list of binning variable names.

		Returns
		-------
		list of str
		"""
		return sorted(self.boundaries.keys())

	def uniqueStr(self):
		"""
		Get a unique string representation of the cell.

		Returns
		-------
		str
		"""
		out = []
		for variable in self.variables():
			variableStr = "{0}={1!r}={2!r}".format(variable, self.boundaries[variable][0], self.boundaries[variable][1])
			out.append(variableStr)
		return ",".join(out)

	@classmethod
	def fromUniqueStr(cls, stringInput):
		"""
		Create a BinningCell from a unique string.

		Parameters
		----------
		stringInput: str

		Returns
		-------
		BinningCell
		"""
		boundaries = {}
		if stringInput:
			variables = stringInput.split(',')
			for variable in variables:
				if variable.count('=') == 2:
					name, lower, upper = variable.split("=")
					boundaries[name] = (float(lower), float(upper))
				else:
					log.error("Cannot get Cell form string '{0}'".format(stringInput))
		return BinningCell(boundaries)

	def overlap(self, other, strict=True):
		"""
		Check if this cell overlaps with another cell in all variables.

		Parameters
		----------
		other : BinningCell
		strict : bool, optional
			If True, overlap is strict.

		Returns
		-------
		bool
		"""
		keys = self.variables()
		if keys != other.variables():
			return False
		for key in keys:
			if not self.overlapInVariable(other, key, strict):
				return False
		return True

	def overlapInVariable(self, other, key, strict=True):
		"""
		Check if this cell overlaps with another cell in a specific variable.

		Parameters
		----------
		other : BinningCell
		key : str
			Variable name.
		strict : bool, optional

		Returns
		-------
		bool
		"""
		def comparator(left, right, direction):
			if direction == ">":
				right, left = left, right
			if strict:
				return left < right
			return left <= right

		if comparator(self.boundaries[key][1], other.boundaries[key][0], "<"):
			return False
		if comparator(self.boundaries[key][0], other.boundaries[key][1], ">"):
			return False
		return True

	def sameBinningVariables(self, other):
		"""
		Check if this cell and another have the same binning variables.

		Parameters
		----------
		other : BinningCell

		Returns
		-------
		bool
		"""
		return sorted(self.boundaries.keys()) == sorted(other.boundaries.keys())


	def inBin(self, binningInfo):
		"""
		Check if binningInfo is in this cell.

		Parameters
		----------
		binningInfo : dict
			Mapping variable names to values.

		Returns
		-------
		bool or numpy.ndarray
			True/False or boolean array for vectorized input.
		"""
		# binningInfo = { "variableName": value }
		isInBin = None
		for variableName, variableBoundaries in self.boundaries.items():
			if variableName in binningInfo:
				value = binningInfo[variableName]
				if isInBin is None:
					isInBin = True if np.isscalar(value) else np.ones(value.size, dtype=bool)
				isInBin &= (value >= variableBoundaries[0])
				isInBin &= (value <  variableBoundaries[1])
			else:
				return False
		return True if isInBin is None else isInBin


	def getSubMultiBin(self, exception=None):
		"""
		Get a sub-cell with specified variables removed.

		Parameters
		----------
		exception : str or list of str, optional

		Returns
		-------
		BinningCell
		"""
		if exception is None:
			return BinningCell(self)

		if isinstance(exception, str):
			exception = [exception]
		boundaries = {}
		for k,values in self.boundaries.items():
			if k not in exception:
				boundaries[k] = values
		return BinningCell(boundaries)

	def getBinCenters(self):
		"""
		Get the centers of all bins.

		Returns
		-------
		dict
			Mapping variable names to bin centers.
		"""
		return {k: 0.5*(v[0]+v[1]) for k,v in self.boundaries.items()}

	def getBinWidths(self):
		"""
		Get the widths of all bins.

		Returns
		-------
		dict
			Mapping variable names to bin widths.
		"""
		return {k: (v[1]-v[0]) for k,v in self.boundaries.items()}

	def toFilename(self):
		"""
		Convert this multibin to a string suitable for file names.

		Returns
		-------
		str
		"""
		return "_".join([ "{0}_{1:.3f}_{2:.3f}".format(var, self.boundaries[var][0], self.boundaries[var][1]) for var in reversed(sorted(self.boundaries.keys())) ])


def binningcell_representer(dumper, data):
	return dumper.represent_mapping('!BinningCell', data.boundaries)

def binningcell_constructor(loader, node):
	boundaries = loader.construct_mapping(node)
	return BinningCell(boundaries)

yaml.add_representer(BinningCell, binningcell_representer)
yaml.add_constructor('!BinningCell', binningcell_constructor)



class Binning:
	"""Class holding binning
	"""
	def __init__(self):
		self._binEdges = {}


	def setBinEdges(self, binEdges: dict[str, np.ndarray]):
		"""Set bin edges

		Args:
			binEdges (dict): Bin edges, where the variable name is the key and the value is an array of bin edges
		"""
		self._binEdges = binEdges


	def setLimits(self, limits: dict[str,tuple[int,float,float]]):
		"""Set bin edges by limits

		Args:
			limits (dict[str,tuple[int,float,float]]): Limits for each variable (key) given also (nBins, upper limit, lower limit)
		"""
		for var, (nbins, lower, upper) in limits.items():
			self._binEdges[var] = np.round(np.array([lower+(upper-lower)/nbins*i for i in range(nbins+1)]),6)


	def validateBin(self, binNumber: int) -> int:
		if not isinstance(binNumber, int):
			raise TypeError('binNumber must be an integer.')
		if not 0 <= binNumber < self.nBins:
			raise Exception(f'The requested bin number {binNumber} is out of range. Valid bin numbers are between 0 and {self.nBins-1}.')
		return binNumber

	def binningCell(self, binNumber: int, variable: str) -> BinningCell:
		bin_lower = self.edges[variable][binNumber]
		bin_upper = self.edges[variable][binNumber + 1]
		return BinningCell({variable: (bin_lower, bin_upper)})

	@property
	def nBins(self) -> int:
		'''The number of bins
		'''
		nBins=0
		for edges in self._binEdges.values():
			nBins+=edges.size-1
		return nBins

	@property
	def edges(self) -> np.array:
		'''The bin edges for each variable
		'''
		return self._binEdges

	@property
	def centers(self) -> np.array:
		'''Returns the bin centers for each variable
		'''
		return {var: 0.5*(edges[1:]+edges[:-1]) for var, edges in self._binEdges.items()}

	@property
	def widths(self) -> np.array:
		'''Returns the bin widths for each variable
		'''
		return {var: edges[1:]-edges[:-1] for var, edges in self._binEdges.items()}
