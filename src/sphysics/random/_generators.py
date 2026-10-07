# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Created on Thursday 17 03 2022
'''

from __future__ import absolute_import, print_function, division

import sys
from contextlib import contextmanager
import numpy as np

from ..utils import Logger

log = Logger("random")


class GeneratorProxy(np.random.Generator):
	'''
	This is a wrapper class for the random number generator.
	It insures, that always the current generator from the Generators object is used to generator random numbers.
	If re-seeding the generators, i.e. initializing new Generator instances with new seeds,
	the GeneratorProxy instance will automatically point to the newly created instance of the Generator, e.g.
	```
	default = generators.default
	default.uniform() # uses the original instance of the default generator
	generators.setSeed(123) # re-seeds, i.e. initializes new instances of all generators
	default.uniform() # used the new instance of the default generator
	```
	'''
	def __init__(self, generators, generatorName) -> None:   # pylint: disable=super-init-not-called
		self._generators = generators
		self._generatorName = generatorName

	def __getattribute__(self, __name: str):
		if __name == '_generators':
			return super().__getattribute__('_generators')
		if __name == '_generatorName':
			return super().__getattribute__('_generatorName')
		if __name == 'genRandomSeed':
			return super().__getattribute__('genRandomSeed')
		return getattr(self._generators._generators[self._generatorName], __name)

	def genRandomSeed(self) -> int:
		"""Generate a random integer that can be used to seed another random number generator

		Returns:
			int: Seed
		"""
		return int(self.integers(2**sys.int_info.bits_per_digit))



class Generators(object):
	'''
	Collection of generators that are used for different sub modules.
	These generators are seeded from a master generator, whose seed can be set with `setSeed`.
	The generators are initialized in a fixed sequence, such that for a given seed of the master generator, the generators are initialized in the same way.
	`setSeed` will re-initialize all generators.
	Per default, the master generator is initialized with a seed from system entropy (see numpy.random docu for seed is None).

	This is a singleton class, i.e. it must not be instanticated explicitly!
	'''

	def __init__(self) -> None:
		self._master = None
		self._defaultAlgorithm = np.random.PCG64

		# list of generators
		self._generatorNames = []
		self._generators = {}
		self._generatorProxys = {}

		self.setSeed(masterSeed=None)

		# IMPORTANT: Keep this order and only append new generators at the end
		self.addGenerator('default')
		self.addGenerator('phaseSpace')
		self.addGenerator('pwaStartparameter')


	@property
	def default(self) -> np.random.Generator:
		return self['default']

	@property
	def phaseSpace(self) -> np.random.Generator:
		return self['phaseSpace']

	@property
	def pwaStartparameter(self) -> np.random.Generator:
		return self['pwaStartparameter']

	def setSeed(self, masterSeed):
		'''
		Reseeds the master generator and all other generators.

		Attention: All other generators will be re-seeded (see description of GeneratorProxy)
		'''
		self._master = np.random.Generator(self._defaultAlgorithm(masterSeed))
		for generatorName in self._generatorNames:
			self._generators[generatorName] = self._newGenerator()

	def addGenerator(self, name) -> np.random.Generator:
		'''
		Add a new generator to the internal list of generators that are managed.
		@param name: Name of the new generator
		'''
		if name in self._generatorNames:
			log.raiseException(Exception, f"Generator with name {name} already exists!")
		self._generatorNames.append(name)
		self._generatorProxys[name] = GeneratorProxy(self, name)
		self._generators[name] = self._newGenerator()
		return self[name]


	def _newGenerator(self, algorithm=None) -> np.random.Generator:
		'''
		Initialize a new random number generator
		@param algorithm: The algorithm that should be used. If non, the default algorithm is used, which is PCG64
		'''
		if algorithm is None:
			algorithm = self._defaultAlgorithm
		seed = self._master.integers(2**sys.int_info.bits_per_digit)
		return np.random.Generator(algorithm(seed))

	def __getitem__(self, generatorName) -> GeneratorProxy:
		return self._generatorProxys[generatorName]


generators = Generators()
default = generators.default



@contextmanager
def setTempSeed(seed: int, generators: Generators = generators):
	"""Re-seed the generators within the context.

	Re-seed the generators, the given one or sphysics.random.generators,
	within the context of a with statement. When leaving the context, the
	generators are reset to their original state.

	Args:
		seed (int): New master seed to re-seed the generators.
		generators (Generators, optional): Generators to re-seed. Defaults to `sphysics.random.generators`.
	"""
	oldMaster = generators._master                 # pylint: disable=protected-access
	oldGenerators = dict(generators._generators)   # pylint: disable=protected-access
	try:
		generators.setSeed(seed)
		yield
	finally:
		generators._master = oldMaster             # pylint: disable=protected-access
		generators._generators = oldGenerators     # pylint: disable=protected-access
