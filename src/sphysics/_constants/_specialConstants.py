# coding: utf-8
'''
Created on Tuesday 14 06 2022
Author: Stefan Wallner
Description: Special constants in addition to those defined in the Constants class
'''
# pylint: disable=invalid-name,protected-access

from __future__ import absolute_import, print_function, division, annotations
import numpy as np

from .._core import constants as core_constants #pylint: disable=import-error


class SpecialConstants:
	def __init__(self) -> None:
		pass


class PhysicalConstants(SpecialConstants):
	def __init__(self) -> None:
		super().__init__()
		# from PDG 2023
		self.c = 299_792_458.0 # speed of light [m/2]
		self.hbar = 1.054571817e-34 # hbar [Js]
		self.hbarc_MeVfm = 197.3269804 # hbar*c [MeV fm]
		self.e = 1.602176634e-19 # fundamental carge [C]
		self.alpha = 1/137.035999084 # fine structure constant



class Amplitudes(SpecialConstants):
	LASS = core_constants.amplitudes.LASS.getInstance()
	FlatteF0 = core_constants.amplitudes.FlatteF0.getInstance()



class CLEOTauPara(SpecialConstants):
	def __init__(self) -> None:
		super().__init__()
		# input parameters
		self.M_rho = 0.774
		self.G_rho = 0.149
		self.M_rho1450 = 1.370
		self.G_rho1450 = 0.386
		self.M_f2	  = 1.275
		self.G_f2	  = 0.185
		self.M_sigma   = 0.860
		self.G_sigma   = 0.880
		self.M_f01370  = 1.186
		self.G_f01370  = 0.350

		# results
		self.M_a1 = 1.331
		self.G_a1 = 0.814
		self.gamma_a1 = 3.32
		self.BKK_a1 = 3.3/100 # 3.3 %


# TauolaBelle Decay Marker
# https://confluence.desy.de/display/BI/Tau+Physics+Analysis+Tools
TauolaBelleMCMode = {
    1:  'e⁻ν ν',
    2:  'μ⁻ν ν',
    3:  'π⁻ ν',
    4:  'ρ⁻ ν',
    5:  'a1 ν',
    6:  'K⁻ ν',
    7:  'K* ν',
    8:  'π⁻π⁻π⁺π⁰ ν',
    9:  'π⁻π⁰π⁰π⁰ ν',
    10: '2π⁻π⁺2π⁰ ν',
    11: '3π⁻2π⁺ ν',
    12: '3π⁻2π⁺π⁰ ν',
    13: '2π⁻π⁺3π⁰ ν',
    14: 'K⁻π⁻K⁺ ν',
    15: 'K⁰π⁻bar(K⁰) ν',
    16: 'K⁻K⁰π⁰ ν',
    17: 'K⁻π⁰π⁰ ν',
    18: 'K⁻π⁻π⁺ ν',
    19: 'π⁻bar(K⁰)π⁰ ν',
    20: 'ηbar(K⁰)π⁰ ν',
    21: 'π⁻π⁰𝛾 ν',
    22: 'K⁻K⁰ ν',
    23: 'π⁻4π⁰ ν',
    24: 'π⁻ωπ⁰ ν',
    25: 'π⁻π⁺π⁻η ν',
    26: 'π⁻π⁰π⁰η ν',
    27: 'K⁻η ν',
    28: 'K*η ν',
    29: 'K⁻π⁺π⁻π⁰ ν',
    30: 'K⁻π⁰π⁰π⁰ ν',
    31: 'K⁰π⁻π⁺π⁻ ν',
    32: 'π⁻bar(K⁰)π⁰π⁰ ν',
    33: 'π⁻K⁺K⁻π⁰ ν',
    34: 'π⁻K⁰bar(K⁶)π⁰ ν',
    35: 'π⁻ωπ⁺π⁻ ν',
    36: 'π⁻ωπ⁰π⁰ ν',
    37: 'e⁻e⁻e⁺ν ν',
    38: 'f1π⁻ ν',
    39: 'K⁻ω ν',
    40: 'K⁻K⁰π⁺π⁻ ν',
    41: 'K⁻K⁰π⁰π⁰ ν',
    42: 'π⁻K⁺bar(K⁰)π⁻ ν',
}

class _TauolaBelle2MCMode(dict):
	def findID(self, name: str) -> int:
		try:
			return next(i for i, i_name in self.items() if i_name == name)
		except StopIteration:
			return None


# TauolaBelle 2 MC decay modes
# from BELLE2-NOTE-PH-2020-055_v2
TauolaBelle2MCMode = _TauolaBelle2MCMode({
	-1 : 'none',
    1  : 'e⁻ν ν',
    2  : 'μ⁻ν ν',
    303: 'π⁻ ν',
    163: 'π⁻π⁰ ν',
    110: 'π⁻π⁰𝛾 ν',
    67 : 'π⁻4π⁰ ν',
    126: 'π⁻π⁰Ks ν',
    127: 'π⁻π⁰KL ν',
    123: 'π⁻KsKL ν',
    135: 'π⁻π⁰𝜂(->π⁻π⁺π⁰) ν',
    124: 'K⁻π⁰Ks ν',
    125: 'K⁻π⁰KL ν',
    112: 'π⁻π⁻π⁺ ν',
    111: 'π⁻π⁰π⁰ ν',
    304: 'K⁻ ν',
    165: 'K⁻π⁰ ν',
    3  : 'π⁻π⁻π⁺π⁰ ν',
    4  : 'π⁻π⁰π⁰π⁰ ν',
    66 : '2π⁻π⁺2π⁰ ν',
    83 : '3π⁻2π⁺ ν',
    84 : '3π⁻2π⁺π⁰ ν',
    85 : '2π⁻π⁺3π⁰ ν',
    103: 'π⁻K⁻K⁺ ν',
    164: 'π⁻K⁰ ν',
    226: 'π⁻Ks ν',
    227: 'π⁻KL ν',
    104: 'π⁻K⁰K⁰ ν',
    106: 'K⁻π⁰π⁰ ν',
    107: 'K⁻π⁻π⁺ ν',
    113: 'K⁻K⁻K⁺ ν',
    114: 'K⁻K⁰K⁰ ν',
    108: 'π⁻K⁰π⁰ ν',
    166: 'K⁻K⁰ ν',
    13 : 'K⁻π⁻π⁺π⁰ ν',
    9  : 'K⁻π⁰π⁰π⁰ ν',
    228: 'K⁻Ks ν',
    229: 'K⁻KL ν',
    23 : 'π⁻K⁻K⁺π⁰ ν',
    24 : 'K⁻K⁻K⁺π⁰ ν',
    172: 'e⁻e⁻e⁺ν ν',
    236: 'π⁻ω(->π⁻π⁺π⁰) ν',
    130: 'π⁻ω(->π⁻π⁺π⁰)π⁰ ν',
    131: 'π⁻ω(->π⁻π⁺)π⁰ ν',
    237: 'π⁻ω(->π⁻π⁺) ν',
    250: 'K⁻ω(->π⁻π⁺π⁰) ν',
    251: 'K⁻ω(->π⁻π⁺) ν',
	32 : '2π⁻π⁺Ks ν',
	33 : '2π⁻π⁺KL ν',
	68 : '3π⁻2π⁺ ν',
})

# TauolaBelle 2 MC decay modes with a single charged track (excluding Ks, i.e. counting as 2 tracks, but including KL, i.e. counting as 0 tracks)
TauolaBelle2MCModesOneProng = [
                  1, 2, 4, 9, 37, 31, 35, 37, 38, 50, 67, 71,
			      106, 110, 111, 122, 125, 127, 132, 133, 134, 137,138, 145, 146, 163, 165,
		          204, 205, 223, 227,229, 238, 241, 242, 243, 246, 247, 252, 257, 258,
		          303, 304, 305, 306, 307, 308, 311, 312, 323, 324, 341, 342, 343, 344, 349,
		          350, 351, 354, 355, 360, 363, 364, 365, 368, 369,]



class BelleII(SpecialConstants):
	def __init__(self) -> None:
		super().__init__()
		self.luminosity_LS1_4S = 361.654  # from https://confluence.desy.de/display/BI/Offline+Luminosity+Page
		self.beam_LER_mom = 4. # GeV/c
		self.beam_corssing = 83e-3 # in rad
		self.beam_HER_mom = 10.5794**2/2/self.beam_LER_mom/(1+np.cos(self.beam_corssing)) # about 7.007 GeV/c

		# form inAcceptance in basf2
		self.detector_theta_ARICH = (14., 30.)
		self.detector_theta_TOP = (32.2, 123.86)
		self.detector_theta_CDC = (17., 150.)
		self.detector_theta_ECL_FWD = (12.4, 31.4)
		self.detector_theta_ECL_Barrel = (32.2, 128.7)
		self.detector_theta_ECL_BWD = (130.7, 155.1)
		self.detector_theta_KLM_FWD = (18., 47.)
		self.detector_theta_KLM_Barrel = (47., 122.)
		self.detector_theta_KLM_BWD = (122, 155.)
		for v in list(self.__dict__.keys()):
			if v.startswith('detector_theta_'):
				theta_range = getattr(self, v)
				cos_theta_range = ( np.cos(np.radians(theta_range[1])), np.cos(np.radians(theta_range[0]))) # pylint: disable=unsubscriptable-object
				new_attr = v.replace('_theta_', '_cosTheta_')
				setattr(self, new_attr, cos_theta_range)
