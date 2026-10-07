'''
:Author: Stefan Wallner
:Description: Created on Nov 20, 2018
'''

from __future__ import absolute_import, division, print_function

import contextlib
import timeit
from collections import OrderedDict, defaultdict
import numpy as np
import tabulate

from ._logger import Logger

log = Logger("timer")

class Timer(object):
	'''
	Time class to measure execution times. This module should be loaded first
	'''


	def __init__(self, name: str=None):
		'''
		Constructor
		'''
		self._times = OrderedDict()
		self._invervalStartTimes = defaultdict(list)
		self._invervalEndTimes = defaultdict(list)
		self._name = name

	def startInverval(self, tag: str):
		"""Add the start time of an interval

		Args:
			tag (str): Tag of the interval
		"""
		if len(self._invervalStartTimes[tag]) != len(self._invervalEndTimes[tag]):
			log.raiseException(ValueError, "Cannot add a start point to interval '{0}', because the end point of the previous iteration is missing!", tag)
		self._invervalStartTimes[tag].append(timeit.default_timer())

	def endInverval(self, tag: str):
		"""Add the end time of an interval

		Args:
			tag (str): Tag of the interval
		"""
		if len(self._invervalStartTimes[tag])-1 != len(self._invervalEndTimes[tag]):
			log.raiseException(ValueError, "Cannot add an end point for interval '{0}', because the start point is missing!", tag)
		self._invervalEndTimes[tag].append(timeit.default_timer())

	@contextlib.contextmanager
	def interval(self, tag: str):
		"""Time the execution time of a code block inside the with statement

		Args:
			tag (str): Tag of the interval
		"""
		try:
			self.startInverval(tag)
			yield self
		finally:
			self.endInverval(tag)

	def time(self, tag=None):
		'''
		Add another time to the timer
		:param tag: Tag for this moment in time
		'''
		if tag is None:
			tag = "tag{0:04d}".format(len(self._times))
		if tag in self._times:
			tag = "{0}{1:04d}".format(tag, len([1 for t in self._times.keys() if tag in t]))

		if tag in self._times:
			log.critical("Cannot add two _times with the same tags!")

		timestamp = timeit.default_timer()

		self._times[tag] = timestamp
		return timestamp


	def printStatistics(self):
		'''Print timing statistics and interval durations.
		'''
		if self._name is None:
			log.info("Timing statistics")
		else:
			log.info("Timing statistics of {0}".format(self._name))
		with log.indented():
			table = self.buildStatisticsTable()
			if table:
				log.info("Durations between time marks:")
				with log.indented():
					for line in table.split('\n'):
						log.info(line)
			table = self.buildIntervalStatisticsTable()
			if table:
				log.info("Interval durations:")
				with log.indented():
					for line in table.split('\n'):
						log.info(line)


	def buildStatistics(self) -> OrderedDict:
		'''Return a dictionary with interval statistics, mapping each tag to the duration of that interval.
		'''
		table = OrderedDict()
		durations = self.durations()
		for k, deltaT in durations.items():
			table[k] = {'duration': deltaT}
		return table

	def buildStatisticsTable(self) -> list:
		'''Return a list of (tag, duration) entries.
		'''
		table = [ (tag, entry['duration']) for tag, entry in self.buildStatistics().items() ]
		return tabulate.tabulate(table,
		                         headers=["Tag", "Duration [s]"],
		                         floatfmt=".3f")


	def buildIntervalStatistics(self) -> OrderedDict:
		"""Build a dictionary with interval statistics, mapping each interval tag to a dictionary with
		(mean, std, min, max, nItterations) of the durations of the interval
		"""
		table = OrderedDict()
		for tag in self._invervalStartTimes.keys():
			if len(self._invervalStartTimes[tag]) != len(self._invervalEndTimes[tag]):
				log.raiseException(ValueError, "End point missing for interval '{0}'", tag)
			durations = np.array(self._invervalEndTimes[tag]) - np.array(self._invervalStartTimes[tag])
			table[tag] = {
			    'mean': np.mean(durations),
			    'std': np.std(durations),
			    'min': np.min(durations),
			    'max': np.max(durations),
			    'itterations': durations.size
			}
		return table

	def buildIntervalStatisticsTable(self) -> str:
		'''Build a table with interval statistics where each row corresponds to (mean, std, min, max, nItterations)
		of the durations of the interval
		'''
		table = [(tag, entry['mean'], entry['std'], entry['min'], entry['max'],
		          entry['itterations'])
		         for tag, entry in self.buildIntervalStatistics().items()]
		return tabulate.tabulate(table,
		                         headers=[
		                             "Tag", "Mean [s]", "Std [s]", "Min [s]",
		                             "Max [s]", "# Iterations"
		                         ],
		                         floatfmt=".3g")


	def durations(self):
		'''Return durations between consecutive tags and the total duration.
		'''
		durations = OrderedDict()
		prev = first = last = None
		for tag, tagTime in self._times.items():
			if prev is not None:
				deltaT = tagTime - self._times[prev]
				durations["{0}->{1}".format(prev, tag)] = deltaT
			else:
				first = tag
			prev = tag
			last = tag
		if first is not None:
			durations["{0}->{1}".format(first, last)] = self._times[last]-self._times[first]
		return durations


def _appendOrderedDict(odict, tag, value):
	# from https://mindthetest.wordpress.com/2016/07/12/append-an-item-to-an-ordereddict/
	root = odict._OrderedDict__root                                        # pylint: disable=protected-access
	last = root[0]
	root[0] = last[1] = odict._OrderedDict__map[tag] = [last, root, tag]   # pylint: disable=protected-access
	dict.__setitem__(odict, tag, value)
