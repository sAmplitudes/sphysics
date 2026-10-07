# coding: utf-8
'''
:Author: Stefan Wallner

:Description: Generic PWA model mase class, Created on Tuesday 15 03 2022
'''

from __future__ import absolute_import, print_function, division

class PWAModel(object):
	'''
	PWA model base class
	'''
	def __init__(self, name: str, description: str = None) -> None:
		self._name = name
		self._description = description if description is not None else ""
		self._waveNames = []

	@property
	def name(self) -> str:
		'''Name of the PWA model
		'''
		return self._name

	@property
	def description(self) -> str:
		''' Returns description of the PWA Model, defaults to empty string
		'''
		return self._description

	@property
	def waveNames(self) -> list:
		'''Returns a list of the wave names
		'''
		return self._waveNames

	@property
	def nWaves(self) -> int:
		'''Returns number of waves
		'''
		return len(self._waveNames)

	def getWaveIndex(self, waveName: str) -> int:
		'''Returns the index at which the specified waveName is stored

		:param waveName: Name of the wave
		:type waveName: str
		'''
		return self._waveNames.index(waveName)
