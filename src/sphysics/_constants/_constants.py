# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Created on Monday 14 03 2022
'''


from __future__ import absolute_import, annotations

from ._constantsClass import ConstantsClass
from ._specialConstants import CLEOTauPara, TauolaBelleMCMode, TauolaBelle2MCMode, TauolaBelle2MCModesOneProng, Amplitudes, BelleII, PhysicalConstants
from ._pdgcodes import PDGCodes
# from ._pdg import PDG
from ..utils import Logger

log = Logger("constants")







Constants = ConstantsClass()

Constants.PhysicsConstants = PhysicalConstants()

Constants.CLEOTauPara = CLEOTauPara()

Constants.TauolaBelleMCMode = TauolaBelleMCMode
Constants.TauolaBelle2MCMode = TauolaBelle2MCMode
Constants.TauolaBelle2MCModesOneProng = TauolaBelle2MCModesOneProng

Constants.PDGCodes = PDGCodes()

Constants.Amplitudes = Amplitudes()

Constants.BelleII = BelleII()

# Constants.PDG = PDG()
