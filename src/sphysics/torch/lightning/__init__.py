# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Lightning modules for training, learning-rate schedulers, and plotting of metrics
'''

#pylint: disable=unused-import

from __future__ import absolute_import

from ._litmodules import LitModuleBase, LitModuleBinaryClassifier, Training
from . import lr_schedulder

try:
	from ._plotting import plotMetric
except ImportError:
	pass
