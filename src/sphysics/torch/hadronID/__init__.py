# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Neural networks for hadron identification, their training datasets, and performance plots
'''

#pylint: disable=unused-import

from __future__ import absolute_import

from ._viewResults import calcEffMissidAndPerformanceHists, calcEffMissid, plot1DPerformance, applyModel
from ._viewResults import plotROC, plotROCRatios
from ._viewResults import plot2DPerformance, plot1DPerformanceSingle, plotPdistributioncosT, plotPdistribution, plotPdistributionMom
from ._network import NeuralNetwork, multipleVariables2Dataset, showModel, variables2Dataset
