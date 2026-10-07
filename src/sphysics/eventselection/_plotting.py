# coding: utf-8
'''
:Author: Stefan Wallner

:Description: Plotting functions for variables, Created on Tuesday 25 04 2023
'''

from __future__ import absolute_import, print_function, division, annotations

from typing import Union, Callable
import numpy as np

from modernplotting import plotter
import modernplotting.mpplot
import modernplotting.specialPlots


from ._variables import Variables
from ._utils import thresholdScan


def plotVariablesDistributions(variables: Variables, variable: Union[str,np.ndarray], style, variableSym: Union[str, np.ndarray]=None,
                               xlim=None, ylim=None, log=False, nbins=None, density=False, channels=None, ncols=5, weights=None):
	'''Plot 1D distribution of the given `variable` for the individual channels.

	:param variables: _description_
	:type variables: Variables
	:param variable: _description_
	:type variable: Union[str,np.ndarray]
	:param style: _description_
	:type style: _type_
	:param variableSym: Symmetrized version of the `variable`. If given, the histograms contain two entries per event
	:type variableSym: Union[str, np.ndarray], optional
	:param xlim: _description_, defaults to None
	:type xlim: _type_, optional
	:param ylim: _description_, defaults to None
	:type ylim: _type_, optional
	:param log: _description_, defaults to False
	:type log: bool, optional
	:param nbins: _description_, defaults to None
	:type nbins: _type_, optional
	:param density: _description_, defaults to False
	:type density: bool, optional
	:param channels: _description_, defaults to None
	:type channels: _type_, optional
	:param ncols: _description_, defaults to 5
	:type ncols: int, optional
	:param weights: _description_, defaults to None
	:type weights: _type_, optional
	:raises NotImplementedError: _description_
	:return: _description_
	:rtype: _type_
	'''
	if isinstance(variable, str):
		variableName = variable
		distributions = variables[variableName]
	else:
		distributions = variable
		try:
			variableName = variables.getNameOfVariable(variable)
		except:
			variableName = ""
	if variableSym is not None:
		if isinstance(variableSym, str):
			variableSymName = variableSym
			distributionsSym = variables[variableSymName]
		else:
			distributionsSym = variableSym
			try:
				variableSymName = variables.getNameOfvariableSym(variableSym)
			except:
				variableSymName = ""
	else:
		distributionsSym = None

	if variableName:
		variableLabel = variableName.replace('_', r'\_') if plotter.rcParams["text.usetex"] else variableName
	else:
		variableLabel = None

	returns = None
	if channels is None:
		channels = variables.getChannels()
	nchannels = len(channels)
	if nchannels == 0:
		pass
	elif nchannels ==1:
		if variableSym is not None:
			raise NotImplementedError()
		plot = style.getPlot1D()
		plot.hist(distributions, range=xlim, density=density, bins=nbins, weights=weights)
		if log:
			plot.setYlog()
		if xlim:
			plot.setXlim(xlim)
		if ylim:
			plot.setYlim(ylim)
		plot.setXticks(6)
		plot.setXlabel(variableLabel)
		returns = plot
	else:
		if 'signal' in channels:
			distributionSignal = distributions[...,variables.idx('signal')]
			if weights is not None:
				distributionSignalWeights = weights[...,variables.idx('signal')]
			else:
				distributionSignalWeights = None
			if distributionsSym is not None:
				distributionSignal = np.concatenate([distributionSignal, distributionsSym[...,variables.idx('signal')]], axis=-1)
				if distributionSignalWeights is not None:
					distributionSignalWeights = np.concatenate([distributionSignalWeights, distributionSignalWeights])

		else:
			distributionSignal = None


		ncols = min(ncols, nchannels-1)
		plots = style.getSubplots1D(nsubplots=nchannels-1, ncols=ncols, sharex=True, titleRight=variableLabel)
		for i, channel in enumerate([ c for c in channels if c != 'signal']):
			distribution = distributions[...,variables.idx(channel)]
			if weights is not None:
				distributionWeights = weights[...,variables.idx(channel)]
			else:
				distributionWeights = None

			if distributionsSym is not None:
				distribution = np.concatenate([distribution, distributionsSym[...,variables.idx(channel)]], axis=-1)
				if distributionWeights is not None:
					distribution = np.concatenate([distributionWeights, distributionWeights], axis=-1)


			plot = plots.getItemRowMajor(i)
			_, binEdges, _ = plot.hist(distribution, label=variables.getChannelLabel(channel), range=xlim, bins=nbins, density=density, weights=distributionWeights)
			if distributionSignal is not None:
				plot.hist(distributionSignal, label='signal', bins=binEdges, density=density, weights=distributionSignalWeights)
			if log:
				plot.setYlog()
			if xlim:
				plot.setXlim(xlim)
			if ylim:
				plot.setYlim(ylim)
			plot.setXticks(6)
			plot.setXgrid()
			plot.setXlabel(variableLabel)
			plot.legend(loc='best')
		returns = plots
	return returns



def compareVariablesDistributions(variables: Variables,
                                  variable: Union[str, np.ndarray],
                                  style: modernplotting.mpplot.PlotterStyle,
                                  variableSym: Union[str, np.ndarray] = None,
                                  xlim: tuple = None,
                                  nbins: Union[int, tuple, np.ndarray] = None,
                                  density: bool = False,
                                  diff: bool = False,
                                  ratio: bool = True,
                                  channels: list = None,
                                  weights: Union[str, np.ndarray] = None) -> modernplotting.mpplot.MPPlot1D:
	'''_summary_

	:param variables: _description_
	:type variables: Variables
	:param variable: _description_
	:type variable: Union[str, np.ndarray]
	:param style: _description_
	:type style: modernplotting.mpplot.PlotterStyle
	:param variableSym: _description_, defaults to None
	:type variableSym: Union[str, np.ndarray], optional
	:param xlim: _description_, defaults to None
	:type xlim: tuple, optional
	:param nbins: _description_, defaults to None
	:type nbins: Union[int, tuple, np.ndarray], optional
	:param density: _description_, defaults to False
	:type density: bool, optional
	:param diff: _description_, defaults to False
	:type diff: bool, optional
	:param ratio: _description_, defaults to True
	:type ratio: bool, optional
	:param channels: _description_, defaults to None
	:type channels: list, optional
	:param weights: _description_, defaults to None
	:type weights: Union[str, np.ndarray], optional
	:return: _description_
	:rtype: modernplotting.mpplot.MPPlot1D
	'''
	if isinstance(variable, str):
		variableName = variable
		distributions = variables[variableName]
	else:
		distributions = variable
		try:
			variableName = variables.getNameOfVariable(variable)
		except:
			variableName = ""
	if variableSym is not None:
		if isinstance(variableSym, str):
			variableSymName = variableSym
			distributionsSym = variables[variableSymName]
		else:
			distributionsSym = variableSym
			try:
				variableSymName = variables.getNameOfvariableSym(variableSym)
			except:
				variableSymName = ""
	else:
		distributionsSym = None

	if isinstance(weights, str):
		weights = variables[weights]

	if variableName:
		variableLabel = variableName.replace(
		    '_', r'\_') if plotter.rcParams["text.usetex"] else variableName
	else:
		variableLabel = None

	if channels is None:
		channels = variables.getChannels()

	distributionList = []
	weightsList = [] if weights is not None else None
	for channel in channels:
		distribution = distributions[..., variables.idx(channel)]
		if weights is not None:
			weight = weights[variables.idx(channel)]
		if distributionsSym is not None:
			distribution = np.concatenate(
			    [distribution, distributionsSym[...,
			                                    variables.idx(channel)]],
			    axis=-1)
			if weights is not None:
				weight = np.concatenate([weight, weight], axis=-1)
		distributionList.append(distribution)
		if weights is not None:
			weightsList.append(weight)

	if diff or ratio:
		plots = modernplotting.specialPlots.compareDistributions(
			style,
			distributionList,
			weights=weightsList,
			labels=[variables.getChannelLabel(c) for c in channels],
			bins=nbins,
			diff=diff,
			hists_kw={"range": xlim, "density": density},
		)
		plots[1, 0].setXticks(6)
		plots[1, 0].setXlabel(variableLabel)
		plots[0, 0].legend(loc='best')
	else:
		plot = style.getPlot1D()
		plot.hists(distributionList, weights=weightsList, range=xlim, bins=nbins, density=density, labels=[variables.getChannelLabel(c) for c in channels])
		plot.setYlim(ymin=0)
		plot.setXlabel(variableLabel)
		plots = plot
	return plots


def plotVariablesDistributions2D(variables: Variables, variableX: Union[str,np.ndarray], variableY: Union[str,np.ndarray],
                                 style, xlim=None, ylim=None, zlim=None, deltaz = None, log=False, nbins=None, channels=None, ncols=5, plots=None,
         weights: Union[str,np.ndarray] = None, ratios=False, diffs=False, **kwargs):
	''' Plot 2D distribution of `variableY` vs `variableX` for the individual channels.

	:param variables: _description_
	:type variables: Variables
	:param variableX: _description_
	:type variableX: Union[str,np.ndarray]
	:param variableY: _description_
	:type variableY: Union[str,np.ndarray]
	:param style: _description_
	:type style: _type_
	:param xlim: _description_, defaults to None
	:type xlim: _type_, optional
	:param ylim: _description_, defaults to None
	:type ylim: _type_, optional
	:param zlim: _description_, defaults to None
	:type zlim: _type_, optional
	:param deltaz: _description_, defaults to None
	:type deltaz: _type_, optional
	:param log: _description_, defaults to False
	:type log: bool, optional
	:param nbins: _description_, defaults to None
	:type nbins: _type_, optional
	:param channels: _description_, defaults to None
	:type channels: _type_, optional
	:param ncols: _description_, defaults to 5
	:type ncols: int, optional
	:param plots: _description_, defaults to None
	:type plots: _type_, optional
	:param weights: _description_, defaults to None
	:type weights: Union[str,np.ndarray], optional
	:param ratios: _description_, defaults to False
	:type ratios: bool, optional
	:param diffs: _description_, defaults to False
	:type diffs: bool, optional
	:return: _description_
	:rtype: _type_
	'''

	if not isinstance(variableX, str):
		distributionsX = variableX
		try:
			variableX = variables.getNameOfVariable(variableX)
		except:
			variableX = ""
	else:
		distributionsX = variables[variableX]

	if not isinstance(variableY, str):
		distributionsY = variableY
		try:
			variableY = variables.getNameOfVariable(variableY)
		except:
			variableY = ""
	else:
		distributionsY = variables[variableY]

	if isinstance(weights, str):
		weights = variables[weights]

	if variableX:
		variableLabelX = variableX.replace('_', r'\_') if plotter.rcParams["text.usetex"] else variableX
	else:
		variableLabelX = None

	if variableY:
		variableLabelY = variableY.replace('_', r'\_') if plotter.rcParams["text.usetex"] else variableY
	else:
		variableLabelY = None

	returns = None

	if channels is None:
		channels = variables.getChannels()

	distributionsXList = []
	distributionsYList = []
	weightsList = []
	for i, channel in enumerate(channels):
		distributionsXList.append(distributionsX[...,variables.idx(channel)])
		distributionsYList.append(distributionsY[...,variables.idx(channel)])
		if weights is not None:
			weightsList.append( weights[...,variables.idx(channel)])
		else:
			weightsList = None

	plots, _, _, _, _ = modernplotting.specialPlots.plot2DHists(style, distributionsXList, distributionsYList, weights=weightsList,
	                                                            rangeX=xlim, rangeY=ylim, bins=nbins, getSubplotsKwargs={'ncols': ncols},
	               ratios=ratios, diffs = diffs, deltaz = deltaz, plots = plots, **kwargs)

	for i in range(len(channels) - (1 if (ratios or diffs) else 0)):
		plot = plots.getItemRowMajor(i)
		if ratios:
			plot.addTitleRight("[{0}] $/$ [{1}]".format(variables.getChannelLabel(channels[i + (1 if (ratios or diffs) else 0)]), channels[0]))
		elif diffs:
			plot.addTitleRight("[{0}] $-$ [{1}]".format(variables.getChannelLabel(channels[i + (1 if (ratios or diffs) else 0)]), channels[0]))
		else:
			plot.addTitleRight(variables.getChannelLabel(channels[i - (1 if (ratios or diffs) else 0)]))
		if log:
			plot.setZlog()
		if zlim is not None:
			plot.setZlim(zlim)
		plot.setZshowColorBar()
		plot.setXticks(6)
		if plots is None or i//plots.nCols == plots.nRows-1: # last row
			plot.setXlabel(variableLabelX)
		plot.setYticks(6)
		if plots is None or i%plots.nCols == 0: # first column
			plot.setYlabel(variableLabelY)
	returns = plots
	return returns

def thresholdScanPlots(
    sample: Variables, cutFunction: Callable, thresholds: np.ndarray,
    style: modernplotting.mpplot.PlotterStyle, weight: str = None,
	efficiencyScaling: float = 1.0,
    plotScanKwargs : dict = None, plotROCKwargs : dict = None
) -> tuple[np.ndarray, np.ndarray, dict[str,np.ndarray], np.ndarray, modernplotting.mpplot.MPPlot1DTwin,
           modernplotting.mpplot.MPPlot1D]:
	"""Perform a threshold scan using weights per default

	Args:
		sample (Variables): Data sample
		cutFunction (function): Cut function with signature (sample, threshold) -> cutMask, which determine the cut mask for a given threshold
		thresholds (np.ndarray): List of thresholds to test (in increasing order)
		style (modernplotting.mpplot.PlotterStyle): Plot style object
		weight (str, optional): Name of variable containing weights. Defaults to 'weight'.
		efficiencyScaling (float, optional): Scale all efficiencies by this factor, i.e. efficiencies and mis-id rates. Default no scaling.
		plotScanKwargs (dict, optional): Kw args for threshold plot `getPlot1DTwin()`
		plotROCKwargs (dict, optional): Kw args for ROC plot `getPlot1D()`

	Returns:
		tuple[np.ndarray, np.ndarray, dict,, np.ndarray modernplotting.mpplot.MPPlot1DTwin, modernplotting.mpplot.MPPlot1D]:
		   - list of thresholds
		   - list of efficiences
		   - dict of lists of impurities for each channel
		   - list of purities, i.e. 1-sum(impurities)
		   - efficiency plot
		   - ROC curve plot
	"""


	thresholds, efficiencies, impurities, purities, missidrates, efficiencyUnc, impurityUnc, purityUnc, missidrateUnc = \
		thresholdScan(sample, cutFunction, thresholds, weight, efficiencyScaling=efficiencyScaling)


	if plotScanKwargs is None:
		plotScanKwargs = {}
	if 'figWidth' not in plotScanKwargs:
		plotScanKwargs['figWidth'] = style.p1dFigSize[0]*2
	plot = style.getPlot1DTwin(**plotScanKwargs)

	plot.plotR(thresholds,
	           efficiencies,
	           ls='--',
	           color=style.colorScheme.black,
	           label='efficiency')
	for imp in impurities:
		plot.plotL(thresholds, impurities[imp], label=sample.getChannelLabel(imp))

	plot.setYlabelR("Efficiency [%]")
	plot.setYlabel("Impurity [%]")
	plot.setYAxesColor(style.p1dDefaultLineColor)
	plot.setYlim(ymin=0)
	plot.setYlimR(ymin=0)
	plot.setYticksR(8)
	plot.setYticks(18)
	plot.setXticks(11)
	plot.setXshowMinorTicks(2)
	plot.setXYgrid(ls='--')
	plot.setXlim((0, 1))
	plot.axesL.legend(loc='upper left')

	if plotROCKwargs is None:
		plotROCKwargs = {}
	roc = style.getPlot1D(**plotROCKwargs)
	roc.plot(purities, efficiencies)
	roc.setXlabel(r"Purity [%]")
	roc.setYlabel(r"Efficiency [%]")
	roc.setYlim((0, 100))
	roc.setXlim(xmax=100)
	roc.setXYgrid(ls='--')
	roc.setXticks(6)
	roc.setYticks(6)
	return thresholds, efficiencies, impurities, purities, missidrates, efficiencyUnc, impurityUnc, purityUnc, missidrateUnc, plot, roc
