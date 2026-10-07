# coding: utf-8
'''
Author: Stefan Wallner
Description: Created on Thursday 04 05 2023
'''
# pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import os
import tensorflow as tf

from ..utils import Logger

from .._core.constants import ParticleDataTable # pylint: disable=import-error

log = Logger("constants")


class MassWidthProxy(object):
	def __init__(self, constants: ConstantsClass):
		self._constants = constants

	def __getitem__(self, name: str) -> float:
		return self._constants.particleMass(name)

	def __getattribute__(self, name: str) -> float:
		if name != '_constants':
			if name.endswith('4S'):
				name = name.replace('4S', '(4S)0')
			if '_' in name:
				isOpen=False
				name = list(name)
				for i,c in enumerate(name):
					if c == '_':
						name[i] = ')' if isOpen else '('
						isOpen = not isOpen
				name = ''.join(name)
			if not name.endswith('0'):
				name += '+'
			if self._constants.particleInTable(name):
				return self[name]
		return object.__getattribute__(self, name)


	def __dir__(self) -> list:
		particles = self._constants.particleData.names
		particles = [p.translate(str.maketrans('','','+-')) for p in particles]
		particles = [p.translate(str.maketrans('()','_'*2)) for p in particles]
		particles = sorted(list(set(particles)))
		return particles


class MassProxy(MassWidthProxy):
	def __getitem__(self, name):
		return self._constants.particleMass(name)

class WidthProxy(MassWidthProxy):
	def __getitem__(self, name):
		return self._constants.particleWidth(name)


class BranchingFractions(dict):
	def __init__(self) -> None:
		super().__init__()
		self['tau-'] = {'1-prong': 0.8523828074, '1-prong w/o Ks': 0.8457913258,
		                'pi-pi-pi+ nu_tau': 0.09306703006, 'pi-pi-pi+ nu_tau w/o K0': 0.09016614895, 'pi-pi-pi+ nu_tau w/o K0 w/o omega': 0.08986788638}

class CrossSections(dict):
	def __init__(self) -> None:
		super().__init__()
		self['e- e+ -> tau- tau+'] = 0.919 # From KKMC


class SpectroscopyMomentumNotation(dict):
	"""Mapping spectroscopy notation S, P, D, ... to integer angular momentum units
	"""
	def __init__(self)->None:
		self['S'] = 0
		self['P'] = 1
		self['D'] = 2
		self['F'] = 3
		self['G'] = 4
		self['H'] = 5
		self['I'] = 6
		self['K'] = 7
		self['L'] = 8

	def __getitem__(self, name: str|int) -> str|int:
		if isinstance(name, int):
			for n, l in self.items():
				if l == name:
					return n
			raise KeyError(f"Cannot find label of angular momentum {name}")
		return super().__getitem__(name)


class ConstantsClass(object):
	"""Hold constants that can be used in the program. Particle properties are read from the particle data table given by the environment variable `SPHYSICS_PARTICLEDATATABLE`.

    Attributes:
        barrierFactorMomentumscale (float): Momentum scale for the barrier factor.
        productionProbReggeAlpha0 (float): alpha_0 of the Regge trajectory used in the production probability model.
        productionProbReggeAlphaPrime (float): alpha' of the Regge trajectory used in the production probability model.
        compassBeamEnergy (float): The beam energy.
        BF (float): Branching fraction.
        crossSections: Cross sections of decays.
        spectroscopyMomentumNotation: Mapping of spectroscopy notation S, P, D, ... to integer angular momentum units.

        PhysicsConstants (object): Instance containing physics constants

            :param c: Speed of light.
            :param hbar: Reduced Planck constant :math:`\hbar` in Js.
            :param hbarc_MeVfm: Reduced Planck constant :math:`\hbar c` in MeVfm.
            :param e: Elementary charge.
            :param alpha: Fine-structure constant.

        CLEOTauPara (object): Instance containing tau decay resonance parameters

			:param M_rho: Mass of :math:`\\rho`.
			:param G_rho: Width of :math:`\\rho`.
			:param M_rho1450: Mass of :math:`\\rho`(1450).
			:param G_rho1450: Width of :math:`\\rho`(1450).
			:param M_f2: Mass of :math:`f_2`.
			:param G_f2: Width of :math:`f_2`.
			:param M_sigma: Mass of :math:`\\sigma`.
			:param G_sigma: Width of :math:`\\sigma`.
			:param M_f01370: Mass of :math:`f_0(1370)`.
			:param G_f01370: Width of :math:`f_0(1370)`.
			:param M_a1: Mass of :math:`a_1`.
			:param G_a1: Width of :math:`a_1`.
			:param gamma_a1: :math:`\\gamma`
			:param BKK_a1: BKK

        TauolaBelleMCMode: Belle tau decay modes.
        TauolaBelle2MCMode: Belle II tau decay modes.
        TauolaBelle2MCModesOneProng: Belle II MC tau decay modes with a single charged track.

        PDGCodes (object): Instance containing PDG codes and names

			:param getPDGCode: Returns the PDG code for a given particle name.
			:param getName: Returns the particle name corresponding to a given PDG code.
			:param __getitem__: Allows lookup using square brackets (e.g., `PDGCodes['pi+']` or `PDGCodes[211]`). Returns the corresponding name or code, or `None` if not found.
			:param __contains__: Returns `True` if a particle name or PDG code exists in the dataset

        BelleII (object): Contains Belle II experiment properties:

			:param luminosity_LS1_4S: Integrated luminosity of the Y(4S) resonance

	"""
	def __init__(self, particleDataFilepath = None):

		self.particleData = ParticleDataTable
		if particleDataFilepath is None:
			if 'SPHYSICS_PARTICLEDATATABLE' in os.environ:
				particleDataFilepath = os.environ['SPHYSICS_PARTICLEDATATABLE']
			else:
				raise Exception("Environment variable `SPHYSICS_PARTICLEDATATABLE`, which should point to the  particle data table file, is not set!")
		if not os.path.isfile(particleDataFilepath):
			raise Exception("Cannot find particle data table '{0}'!".format(particleDataFilepath))
		self.particleData.readFile(particleDataFilepath)


		self.barrierFactorMomentumscale = 0.1973 # momentum scale 0.1973 GeV/c corresponds to 1 fm interaction radius

		self.productionProbReggeAlpha0     = 1.2  # alpha_0 of regge trajectory as used in the production probability model
		self.productionProbReggeAlphaPrime = 0.26 # alpha' of regge trajectory as used in the production probability model

		self.compassBeamEnergy = 191.0

		self.BF = BranchingFractions()
		self.crossSections = CrossSections()

		self.spectroscopyMomentumNotation = SpectroscopyMomentumNotation()




	def particleInTable(self, name):
		'''Checks if particle `name` is in `ParticleDataTable`

		:param name: Name of the particle
		:type name: str
		:return: True if particle is found the table, False otherwise
		:rtype: bool
		'''
		return self.particleData.isInTable(name)

	def particleMass(self, name):
		'''Returns particle mass that is specified in `particleDataTable`.

		:param name: Name of the particle
		:type name: str
		'''
		return self.particleData.entry(name).mass

	def particleWidth(self, name):
		'''Returns particle width that is specified in `particleDataTable`.

		:param name: Name of the particle
		:type name: str
		'''
		return self.particleData.entry(name).width

	@property
	def M(self):
		return MassProxy(self)

	@property
	def G(self):                   #pylint: disable=invalid-name
		return WidthProxy(self)


	def particleMassTf(self, name):
		'''Return mass as a TensorFlow constant of the particle `name`.

		:param name: Name of the particle
		:type name: str
		'''
		with tf.name_scope("particleMasses/"):
			varName = name.replace('(', '_').replace(')', '_').replace('+', 'p')
			massTf = tf.constant(self.particleMass(name), name=varName, dtype=tf.float64)
		return massTf
