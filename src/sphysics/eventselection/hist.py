# coding: utf-8
'''
:Author: Stefan Wallner

:Description: Histogramming classes and functions, Created on Tuesday 25 04 2023
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np


from .. import math
from ..utils import Logger

log = Logger("eventselection")

class SparseHist:
	"""
	Generates a sparse histogram from provided value array and (optional) weights.

	The bin index, i.e. :math:`\\frac{x - \\text{refBinEdge}}{\\text{binWidth}}` must still be representable by a 64-bit integer.
	"""
	def __init__(self, binWidth: float, refBinEdge: float) -> None:
		self._counts = {}
		self._binWidth = binWidth
		self._refBinEdge = refBinEdge

	def fill(self, values: np.ndarray, weights: np.ndarray = None) -> None:
		"""Fills the values into the histogram and weights the data if weights are provided.

		:param values: Array of values to be filled into the sparese histogram
		:param weights: Weights used to fill the histogram. Defaults to None, i.e. weight 1 per entry.
		"""
		if weights is not None and weights.size != values.size:
			log.raiseException(ValueError, f"Values hase size {values.size}, while weights have a different size of {weights.size}!")
		binIndicesOfValues = np.floor((values-self._refBinEdge)/self._binWidth).astype(np.int32)
		binIndicesOfCounts, counts = np.unique(binIndicesOfValues, return_counts=True)
		for binIndex, count in zip(binIndicesOfCounts, counts):
			if weights is not None:
				count = math.sum(weights[np.argwhere(binIndicesOfValues==binIndex)])
			if binIndex in self._counts:
				self._counts[binIndex] += count
			else:
				self._counts[binIndex] = count
	@property
	def counts(self) -> dict:
		''' Returns the counts of each bin in the histogram and stores it in a dictionary
		'''
		return self._counts
	@property
	def binCenters(self) -> dict:
		''' Returns the centre of each bin in the histogram and stores it in a dictionary
		'''
		return {i: self._refBinEdge + (i+0.5)*self._binWidth for i in self._counts}
