# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Created on Monday 14 03 2022
'''


import pdg
import pdg.data

from ..utils import Logger

log = Logger("constants")


class PDG:
	def __init__(self):
		self.api = pdg.connect()


	def get(self, pdgid: str, edition: str|None=None) -> pdg.data.PdgData:
		"""
		Get the PDG data for a given PDG ID.

		:param pdgid: The PDG ID of the particle.
		:param edition: The edition of the PDG data to use. If None, the latest edition is used.
		:return: The PDG data for the given PDG ID.
		"""
		return self.api.get(pdgid, edition=edition)


	def printInfo(self):
		log.info(f"Loaded PDG version {self.api.info('edition')}")
		log.info(f"	{self.api.info('citation')}")
