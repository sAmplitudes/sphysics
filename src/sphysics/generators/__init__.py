# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Phase-space generators for two-body and n-body decays and for tau decays
'''

#pylint: disable=unused-import

from __future__ import absolute_import

from ._phaseSpace import twoBodyDecay, NBodyGenerator
from ._tau import tauToThreePi, tauToNBody
