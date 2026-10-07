# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Utils for PWA fits
'''

from typing import Optional
from pathlib import Path
import yaml

from ...utils import Logger
from ._binning import BinningCell

log = Logger('PWA')


class WaveSet:
	"""
	Class to hold a set of waves.
	"""

	def __init__(self):
		"""
		Initialize an empty WaveSet.
		"""
		self._waves: list[str] = []
		self._waveRanges: dict[str, list[BinningCell]] = {}
		self._referenceWaves: list[dict] = []

	def __getitem__(self, key):
		"""
		Get wave by index.

		Parameters
		----------
		key : int

		Returns
		-------
		str
		"""
		return self._waves[key]

	def __len__(self):
		"""
		Get the number of waves.

		Returns
		-------
		int
		"""
		return len(self._waves)

	def __iter__(self):
		"""
		Iterate over waves.

		Returns
		-------
		iterator
		"""
		return iter(self._waves)

	@property
	def waves(self):
		"""
		Get the list of wave names.

		Returns
		-------
		list of str
		"""
		return self._waves

	@property
	def waveRanges(self):
		"""
		Get the dictionary of wave ranges.

		Returns
		-------
		dict
		"""
		return self._waveRanges

	def getWaveRanges(self, waveName: str) -> list[BinningCell]:
		"""
		Get the list of BinningCells for a wave.

		Parameters
		----------
		waveName : str

		Returns
		-------
		list of BinningCell or None
		"""
		if waveName in self._waveRanges:
			return self._waveRanges[waveName]
		return None

	def getReferenceWave(self, cell: BinningCell = None) -> str:
		"""
		Get the reference wave for a given cell.

		Parameters
		----------
		cell : BinningCell

		Returns
		-------
		str or None
		"""
		if cell is None:
			if len(self._referenceWaves) != 1:
				log.warning("Multiple reference waves exist, but no cell was provided. Returning None.")
				return None
			return self._referenceWaves[0]['name']
		for ref in self._referenceWaves:
			if 'ranges' not in ref:
				return ref['name']
			if any([range_.overlap(cell, strict=False) for range_ in ref['ranges']]):
				return ref['name']
		return None

	def addWave(self, waveName: str, waveRanges: Optional[list[BinningCell]] = None):
		"""
		Add a wave to the set.

		Parameters
		----------
		waveName : str
		waveRanges : list of BinningCell or BinningCell, optional
		"""
		if waveName in self._waves:
			log.raiseException(ValueError, f"Wave '{waveName}' already exists in the set. Overwriting.")
		self._waves.append(waveName)
		if waveRanges is None:
			pass
		elif isinstance(waveRanges, BinningCell):
			self._waveRanges[waveName] = [waveRanges]
		elif isinstance(waveRanges, list):
			self._waveRanges[waveName] = waveRanges
		else:
			log.raiseException(TypeError, f"Wave ranges must be a list of BinningCell objects, got {type(waveRanges)}.")


	def addReferenceWave(self, waveName: str, ranges: Optional[list[BinningCell]] = None):
		"""
		Add a reference wave.

		Parameters
		----------
		waveName : str
		ranges : list of BinningCell or BinningCell, optional
		"""
		if ranges is not None:
			if not isinstance(ranges, list):
				ranges = [ranges]
			for range_ in ranges:
				if self.getReferenceWave(range_) is not None:
					log.raiseException(ValueError, f"Reference wave '{waveName}' already exists for one of the given ranges.")
		elif self._referenceWaves:
			log.raiseException(ValueError, f"Global reference wave '{waveName}' cannot be added because another reference wave already exists.")
		data = {'name': waveName}
		if ranges is not None:
			data['ranges'] = list(ranges)
		self._referenceWaves.append(data)


	def __str__(self):
		"""
		Return a string representation of the WaveSet.

		Returns
		-------
		str
		"""
		return f'WaveSet({self._waves})'


	def toYaml(self, yamlPath: Path):
		"""
		Write the WaveSet to a YAML file.

		Parameters
		----------
		yamlPath : Path
		"""
		data = {
			'wavelist': [],
			'referenceWaves': self._referenceWaves
		}

		for wave in self.waves:
			waveData = {'name': wave}
			ranges_ = self.getWaveRanges(wave)
			if ranges_ is not None:
				waveData['ranges'] = ranges_
			data['wavelist'].append(waveData)

		with open(yamlPath, 'w', encoding='utf-8') as yamlFile:
			yaml.dump(data, yamlFile, default_flow_style=False)
			log.debug(f"Waves written to {yamlPath}")

	@classmethod
	def fromYaml(cls, yamlPath: Path):
		"""
		Load a WaveSet from a YAML file.

		Parameters
		----------
		yamlPath : Path

		Returns
		-------
		WaveSet
		"""
		waveSet = cls()
		with open(yamlPath, 'r', encoding='utf-8') as yamlFile:
			data = yaml.full_load(yamlFile)
			if 'wavelist' not in data:
				log.raiseException(ValueError, f"YAML file {yamlPath} does not contain 'wavelist'.")
			for waveData in data['wavelist']:
				waveName = waveData['name']
				ranges_ = waveData.get('ranges', None)
				if ranges_ is not None:
					ranges_ = [BinningCell(r) for r in ranges_]
				waveSet.addWave(waveName, ranges_)
			if 'referenceWaves' in data:
				for ref in data['referenceWaves']:
					waveName = ref['name']
					ranges_ = ref.get('ranges', None)
					if ranges_ is not None:
						ranges_ = [BinningCell(r) for r in ranges_]
					waveSet.addReferenceWave(waveName, ranges_)
		return waveSet


	def getWavelist(self, cell: Optional[BinningCell] = None) -> list[str]:
		"""
		Get the list of waves for a given cell.

		Parameters
		----------
		cell : BinningCell, optional

		Returns
		-------
		list of str
		"""
		wavelist = []
		for wave in self.waves:
			if cell is None or self.getWaveRanges(wave) is None:
				wavelist.append(wave)
			elif any([cell in range_ for range_ in self.getWaveRanges(wave)]):
				wavelist.append(wave)
		return wavelist
