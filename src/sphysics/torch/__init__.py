# coding: utf-8
'''
:Description: Training of PyTorch models, metrics, hadron identification, and Lightning modules
'''

#pylint: disable=unused-import

from __future__ import absolute_import

from ._training import TrainingHistory, Trainer,saveBestEpoch
from . import metrics
from . import hadronID
from . import lightning
try:
	from ._plottingUtils import plotSequentialModel
except ImportError:
	pass
