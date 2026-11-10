# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Partial-wave analysis: models for two-body and three-body decays, fits, barrier factors, polarimeters, and utilities
'''

#pylint: disable=unused-import

from __future__ import absolute_import

from . import threebody, twobody, utils, fit, polarimeter

from ._model import PWAModel
