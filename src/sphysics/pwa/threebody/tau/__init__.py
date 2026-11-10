# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Models, kinematics, and event handling for tau decays into three charged pions
'''

#pylint: disable=unused-import

from __future__ import absolute_import

from ._model import Tau2ThreeCharedPi
from ._kinematics import tau3Kinematics, KinematicsMCT, getKuhnEulerAngles, getTauMomentumComponentsCMS
from ._kinematics import calcTauHelicityCosTheta_CMS, calcTauHelicityCosTheta_tauRF
from ._utils import EventsTau2ThreeChargedPi, variablesReco2Events, variablesMCT2Events
from ._utils import calcDecayAmplitudesIntegratedNormIntegrals, setDecayAmplitudesIntegratedNormIntegrals, calcIntegralMatrix
