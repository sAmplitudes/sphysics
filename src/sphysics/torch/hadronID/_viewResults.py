# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Functions to view the result of a notebook, Created on Thursday 01 12 2022
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import numpy.polynomial
from numba import jit, prange
import scipy.interpolate
import torch

from .._training import Trainer
from ._network import multipleVariables2Dataset
from ... import math
from ...utils import Logger

log = Logger("torch")



def getIsSpeciesAsHypothesis(species, hypothesis, isSpecies, probRatios, thresholds, classMap):
	return isSpecies[species][:,None] & ( probRatios[...,classMap[hypothesis]][:,None] > thresholds[None,:] )

@jit(nopython=True, parallel=True)
def nIsSpeciesAsHypothesis(sWeights, isSpecies, probRatios, thresholds):
	sums = np.zeros(thresholds.shape[0])
	squaredSums = np.zeros(thresholds.shape[0])
	for iThreshold in prange(sums.shape[0]):  #pylint: disable=not-an-iterable
		for i in range(sWeights.shape[0]):
			if isSpecies[i] & (probRatios[i] > thresholds[iThreshold]):
				sums[iThreshold] += sWeights[i]
				squaredSums[iThreshold] += sWeights[i]**2
	return sums, squaredSums


def calcEvetnNumbers(species, hypothesis, isSpecies, probRatios, thresholds,
                     sWeights, classMap):

	nSpeciesAsHyphothesis, nSpeciesAsHyphothesisSsquared = nIsSpeciesAsHypothesis(
	    sWeights.cpu().numpy(), isSpecies[species].cpu().numpy(),
	    probRatios[..., classMap[hypothesis]].cpu().numpy(),
	    thresholds.cpu().numpy())
	nSpecies = torch.sum(sWeights[isSpecies[species]]).cpu().numpy()
	nSpeciesSsquared = torch.sum(sWeights[isSpecies[species]]**2).cpu().numpy()
	return nSpeciesAsHyphothesis, nSpeciesAsHyphothesisSsquared, nSpecies, nSpeciesSsquared


def calcEffSH(species, hypothesis, isSpecies, probRatios, threshold, sWeights,
              classMap):
	nSpeciesAsHyphothesis, _, nSpecies, _ = calcEvetnNumbers(
	    species, hypothesis, isSpecies, probRatios, threshold, sWeights,
	    classMap)
	return nSpeciesAsHyphothesis / nSpecies


def calcEffMissid(model,
                  nThresholds=2000):
	"""Calculate the efficiencies and missedintification rates.

	Args:
		model: Model to evaluate
		fixedMissID (float, optional): Determine the threshold of this misidentification rate for the species in `missIDSpecies`
		                                and create 1D and 2D performance plots for this threshold
		nThresholds (int, optional): Number of thresholds for which the efficiencies and miss-ids are determined. Defaults to 2000.
	"""
	sWeights = model.datasetTestAux[..., -1]
	if not hasattr(model, 'thresholds') or model.thresholds.size < nThresholds:
		model.thresholds = np.linspace(0, 1, nThresholds)

		model.efficiencies = {}
		model.missids = {}
		model.isSpecies = {}
		for s in ['K', 'pi']:
			model.isSpecies[s] = model.datasetTestLabels == model.classMap[s]
			model.efficiencies[s] = []
			model.missids[s] = {}
			for h in filter(lambda h, s=s: h != s, ['K', 'pi']):
				model.missids[s][h] = []

		for s in model.efficiencies.keys():
			model.efficiencies[s] = calcEffSH(s, s, model.isSpecies,
			                                  model.probRatios,
			                                  torch.tensor(model.thresholds),
			                                  sWeights, model.classMap)
			for h in model.missids[s].keys():
				model.missids[s][h] = calcEffSH(s, h, model.isSpecies,
				                                model.probRatios,
				                                torch.tensor(model.thresholds),
				                                sWeights, model.classMap)
		model.aurocs = {}
		for s in model.efficiencies.keys():
			model.aurocs[s] = {}
			for missidHypothesis in model.missids[s].keys():
				model.aurocs[s][missidHypothesis] = calcAUROC(model,
				                                species=s,
				                                missidSpecies=missidHypothesis)


def calcEffMissidAndPerformanceHists(model,
                  binEdges,
                  fixedMissID=2e-2,
                  missIDSpecies=('pi', 'K'),
                  quiet=False,
                  nThresholds=2000):
	"""Calculate the efficiencies, missedintification ranges and 1D and 2D performance bins

	Args:
		model: Model to evaluate
		binEdges (optional): The bin edges for the performance histograms
		fixedMissID (float, optional): Determine the threshold of this misidentification rate for the species in `missIDSpecies`
		                                and create 1D and 2D performance plots for this threshold
		missIDSpecies (tuple, optional): Determine threshold for misidentification rate for particle [0] as particle [1]. Defaults to ('pi', 'K').
		quiet (bool, optional): Do not print information. Defaults to False.
		nThresholds (int, optional): Number of thresholds for which the efficiencies and miss-ids are determined. Defaults to 2000.
	"""
	calcEffMissid(model, nThresholds)
	sWeights = model.datasetTestAux[..., -1]
	# find threshold for fixed miss-ID
	threshold = model.thresholds[np.argmin(
	    np.abs(
	        np.array(model.missids[missIDSpecies[0]][missIDSpecies[1]]) -
	        fixedMissID))]
	if not quiet:
		log.info("Set threshold for {0:40s} to {1}".format(model.projectTag, threshold))

	isSpeciesAsHypothesis = {}
	model.histsSpecies = {}
	model.histsSpeciesSquared = {}
	model.histsSpeciesAsHypothesis = {}
	model.histsSpeciesAsHypothesisSquared = {}
	for s in ['K', 'pi']:
		model.histsSpecies[s], _ = np.histogramdd(
		    model.datasetTestAux[model.isSpecies[s], :3].cpu().numpy(),
		    bins=binEdges,
		    weights=model.datasetTestAux[model.isSpecies[s], -1].cpu().numpy())
		model.histsSpeciesSquared[s], _ = np.histogramdd(
		    model.datasetTestAux[model.isSpecies[s], :3].cpu().numpy(),
		    bins=binEdges,
		    weights=model.datasetTestAux[model.isSpecies[s],
		                                 -1].cpu().numpy()**2)
		isSpeciesAsHypothesis[s] = {}
		model.histsSpeciesAsHypothesis[s] = {}
		model.histsSpeciesAsHypothesisSquared[s] = {}
		for h in ['K', 'pi']:
			isSpeciesAsHypothesis[s][h] = getIsSpeciesAsHypothesis(
			    s, h, model.isSpecies, model.probRatios,
			    torch.tensor(np.array([threshold])), model.classMap)[:, 0]

			model.histsSpeciesAsHypothesis[s][h], _ = np.histogramdd(
			    model.datasetTestAux[
			        isSpeciesAsHypothesis[s][h], :3].cpu().numpy(),
			    bins=binEdges,
			    weights=sWeights[isSpeciesAsHypothesis[s][h]].cpu().numpy())
			model.histsSpeciesAsHypothesisSquared[s][h], _ = np.histogramdd(
			    model.datasetTestAux[
			        isSpeciesAsHypothesis[s][h], :3].cpu().numpy(),
			    bins=binEdges,
			    weights=sWeights[isSpeciesAsHypothesis[s][h]].cpu().numpy()**2)

	return model



def loadCheckpoint(projectDir,
                   variables,
                   checkpoint=None,
                   bestOfLast=None,
                   verbose=True,
       device=None):

	model = Trainer.loadCheckpoint(projectDir, checkpoint=checkpoint, bestOfLast=bestOfLast, verbose=verbose, device=device).model
	applyModel(model, variables, device=device)
	return model


def applyModel(model, variablesToUse, device=None):
	'''Apply trained model to test data (sets model to evaluation mode) and compute predicted (log-)probabilities, log-probability differences and
	probability ratios.

	:param model: Model to evaluate
	:param variablesToUse: Mapping from particle labels to list of input variables
	:type variablesToUse: dict
	:param device: Torch device to use (for example 'cpu' or 'cuda'), defaults to None
	:type device: torch.device, optional
	'''
	if model.auxVariables is model.inputVariables:  # fix bug in optuna training function
		model.auxVariables = [
		    'p', 'cosTheta', 'phi', 'globalR_Kpi', 'sWeights'
		]
	model.auxVariables = [
	    'p', 'cosTheta', 'phi', 'globalR_Kpi', 'K_ID', 'pi_ID', 'sWeights'
	]
	auxVariables = list(model.auxVariables)
	if auxVariables[-1] != 'sWeights':
		auxVariables.append('sWeights')
	auxVariables = [name.replace(r'DST_D0_{p}_', '') for name in auxVariables]
	inputVariables = [
	    name.replace(r'DST_D0_{p}_', '') for name in model.inputVariables
	]
	labels = []
	multipleVariables = []
	for s in ['K', 'pi']:
		labels.append(model.classMap[s])
		multipleVariables.append(variablesToUse[s])
	datasetTest = multipleVariables2Dataset(labels, multipleVariables,
	                                        inputVariables, auxVariables,
	                                        device)
	model.datasetTestInput, model.datasetTestLabels, model.datasetTestAux = datasetTest.tensors #pylint: disable=unbalanced-tuple-unpacking
	model.eval()
	with torch.no_grad():
		model.predictedLogProb = model(model.datasetTestInput)
		model.predictedProb = torch.exp(model.predictedLogProb)
	model.logProbDiff = model.predictedLogProb[:,
	                                           0] - model.predictedLogProb[:,
	                                                                       1]
	model.probRatios = model.predictedProb / torch.sum(model.predictedProb,
	                                                   axis=1).reshape(-1, 1)

	model.datasetTestLabelsNp = model.datasetTestLabels.cpu().numpy()
	model.datasetTestAuxNp = model.datasetTestAux.cpu().numpy()
	model.predictedLogProbNp = model.predictedLogProb.cpu().numpy()
	model.predictedProbNp = model.predictedProb.cpu().numpy()
	model.logProbDiffNp = model.logProbDiff.cpu().numpy()
	model.probRatiosNp = model.probRatios.cpu().numpy()

	model.label = model.projectTag.replace('_', r'\_')


def calcAUROC(m, species='K', missidSpecies='pi'):
	auroc = None
	# log.info("AUROC:")
	with log.indented():
		# x = np.linspace(0, 1, 1000)
		order = 50
		x, xWeights = numpy.polynomial.legendre.leggauss(order)
		x = 0.5 * (x + 1.)

		missids = np.array(m.missids[missidSpecies][species])
		efficiencies = np.array(m.efficiencies[species])
		iSorted = np.argsort(missids)
		missids = missids[iSorted]
		efficiencies = efficiencies[iSorted]

		def interpolate(x):

			iwhere = np.argwhere(
			    (missids.reshape(-1, 1) - x.reshape(1, -1)) > 0)
			iFirstAbove = np.empty_like(x, dtype=int)
			for i in range(x.size):
				if i in iwhere[..., 1]:
					iFirstAbove[i] = np.min(iwhere[iwhere[..., 1] == i][...,0])
				else:
					iFirstAbove[i] = -1

			inRange = np.logical_and(x > missids[0], x <= missids[-1])
			belowRange = x <= missids[0]
			aboveRange = x > missids[-1]

			iFirstAbove = iFirstAbove[inRange]
			iFirstBelow = iFirstAbove - 1

			y = np.empty_like(x)

			y[inRange] = (x[inRange] - missids[iFirstBelow]) / (
			    missids[iFirstAbove] -
			    missids[iFirstBelow]) * efficiencies[iFirstBelow] + (
			        missids[iFirstAbove] - x[inRange]) / (
			            missids[iFirstAbove] -
			            missids[iFirstBelow]) * efficiencies[iFirstAbove]
			y[belowRange] = efficiencies[0]
			y[aboveRange] = efficiencies[-1]

			return y

		y = interpolate(x)
		# y = np.interp(x, missids, efficiencies)

		auroc = np.sum(y * xWeights) / 2.

		# def f(x):
		#     return interpolate(np.array(x).reshape(1))
		# scipy.integrate.quad_vec(f, 0.0, 1.0)

	return auroc


def calcEfficiencyUncertainty(nSignal, nTotal, uncSignalSquared,
                              uncTotalSquared):
	ratio = math.divide(nSignal,
	                    nTotal,
	                    where=nTotal > 0,
	                    whereNotValue=np.nan)
	oneOverTotal = math.divide(np.ones_like(nTotal),
	                           nTotal,
	                           where=nTotal > 0,
	                           whereNotValue=np.nan)
	uncertaintySquared = oneOverTotal**2 * (
	    (1. - 2. * ratio) * uncSignalSquared + (ratio)**2 * uncTotalSquared)
	if math.min(uncertaintySquared) < -1e-10:
		log.raiseException(
		    ValueError,
		    f"Uncertainty-squared smaller than 0: {uncertaintySquared}")
	return math.sqrt(math.abs(uncertaintySquared))


def plot1DPerformanceSingle(fixedMissID,
                            models2compare,
                            binEdges,
                            binningVariableLabels,
                            style,
                            combination='K_pi',
                            legend=True,
                            yrange=None):
	'''Create 1D efficiency plot of one or multiple models for all variables in 'binningVariableLabels'.

	:param fixedMissID: Threshold of the misidentification rate
	:type fixedMissID: float
	:param models2compare: List of models to compare
	:type models2compare: list
	:param binEdges: The bin edges for the performance histograms
	:param binningVariableLabels: The labels for the variables
	:type binningVariableLabels: list of str
	:param style: Object containing plotting methods, setting style of the plot
	:param combination: Combination of particles to plot, currently only 'K_pi' implemented
	:type combination: str, optional
	:param legend: Boolean which determines whether a legend is plotted, defaults to True
	:type legend: bool, optional
	:param yrange: Optional limit for the range of the y-axis, defaults to None
	:type yrange: tuple of float, optional
	'''

	if combination == 'K_pi':
		species, hypothesis, missIDSpecies, rateLabel, rateRange = (
		    'K', 'K', 'pi', r'$K$ Efficiency [\si\percent]', (0.0, 105))
	else:
		raise Exception(f"Unknown combination '{combination}'")
	if yrange is not None:
		rateRange = yrange

	iVar = 0

	for m in models2compare:
		calcEffMissidAndPerformanceHists(m, binEdges,
		              fixedMissID=fixedMissID,
		              missIDSpecies=(missIDSpecies, hypothesis))

	plot = style.getPlot1D()

	summedAxes = tuple(i for i in range(3) if i != iVar)
	for m in models2compare:
		label = m.label
		total = np.sum(m.histsSpecies[species], axis=summedAxes)
		identified = np.sum(m.histsSpeciesAsHypothesis[species][hypothesis],
		                    axis=summedAxes)
		totalUncSquared = np.sum(m.histsSpeciesSquared[species],
		                         axis=summedAxes)
		identifiedUncSquared = np.sum(
		    m.histsSpeciesAsHypothesisSquared[species][hypothesis],
		    axis=summedAxes)
		rate = np.empty_like(total)
		np.divide(identified, total, out=rate, where=total > 0)
		rateUnc = calcEfficiencyUncertainty(identified, total,
		                                    identifiedUncSquared,
		                                    totalUncSquared)
		plot.plotErrorBar(0.5 * (binEdges[iVar][1:] + binEdges[iVar][:-1]),
		                  rate * 100,
		                  yerr=rateUnc * 100,
		                  label=label)

	plot.setYlim(rateRange)
	plot.setXYgrid()
	plot.setXticks(10)
	plot.setYticks(6)
	plot.setXlabel(binningVariableLabels[iVar])
	plot.setYlabel(rateLabel)
	if legend:
		plot.legend(loc='best' if legend is True else legend,
		            fontsize=style.legendFontSize * 0.8)
	plot.finish()
	return plot


def _calcRateAndRateUncertainty(m, species, hypothesis, summedAxes):
	total = np.sum(m.histsSpecies[species], axis=summedAxes)
	identified = np.sum(
	 m.histsSpeciesAsHypothesis[species][hypothesis],
	 axis=summedAxes)
	totalUncSquared = np.sum(m.histsSpeciesSquared[species],
	       axis=summedAxes)
	identifiedUncSquared = np.sum(
	 m.histsSpeciesAsHypothesisSquared[species][hypothesis],
	 axis=summedAxes)
	rate = np.empty_like(total)
	np.divide(identified, total, out=rate, where=total > 0)
	rateUnc = calcEfficiencyUncertainty(identified, total,
	         identifiedUncSquared,
	         totalUncSquared)
	return rate, rateUnc

def plot1DPerformance(fixedMissID,
                      missIDyMax,
                      models2compare,
                      binEdges,
                      binningVariableLabels,
                      style,
                      plotVars=('p', 'cosTheta', 'phi'),
                      combinations='K_pi',
                      legend=True,
       reference4Ratio=None,
       ylimit4Ratio=None,
    datasetLabel=None):
	'''Create 1D misidentification and efficiency plots of one or multiple models.

	:param fixedMissID: Threshold of the misidentification rate
	:type fixedMissID: float
	:param missIDyMax: Maximum value for misidentification rate
	:type missIDyMax: float
	:param models2compare: List of models to compare
	:type models2compare: list
	:param binEdges: The bin edges for the performance histograms
	:param binningVariableLabels: The labels for the variables
	:type binningVariableLabels: list of str
	:param style: Object containing plotting methods, setting style of the plot
	:param plotVars: Variables that are plotted, defaults to ('p', 'cosTheta', 'phi')
	:type plotVars: tuple, optional
	:param combinations: Combination of particles to plot, currently only 'K_pi' and 'pi_K' implemented
	:type combinations: str, optional
	:param legend: Boolean which determines whether a legend is plotted, defaults to True
	:type legend: bool, optional
	:param reference4Ratio: If set, plot ratio w.r.t this reference, defaults to None
	:type reference4Ratio: int or None, optional
	:param ylimit4Ratio: Limit y-axis when plotting ratios, defaults to None
	:type ylimit4Ratio: tuple, optional
	:param datasetLabel: Title that is added to the plot, defaults to None
	:type datasetLabel: str, optional
	'''

	if combinations == 'K_pi':
		combinations = [
		    ('K', 'K', r'$K$ Efficiency', (0.0, 105)),
		    ('pi', 'K', r'$\pi$ Miss-ID', (0.0, missIDyMax)),
		]
	elif combinations == 'pi_K':
		combinations = [
		    ('pi', 'pi', r'$\pi$ Efficiency', (0.0, 105)),
		    ('K', 'pi', r'$K$ Miss-ID', (0.0, missIDyMax)),
		]

	else:
		raise Exception(f"Unknown combination '{combinations}'")

	for m in models2compare:
		calcEffMissidAndPerformanceHists(m, binEdges,
		              fixedMissID=fixedMissID,
		              missIDSpecies=(combinations[1][0], combinations[1][1]))

	legendOffset = 1 if legend is True else False
	plots = style.getSubplots1D(nrows=len(plotVars) + legendOffset,
	                            ncols=len(combinations),
	                            figWidth=8,
	                            figHeight=2+2*len(plotVars))
	plots[0, 0].addTitleLeft(
	    r"Fixed average miss-ID rate: \SI{{{0:.1f}}}{{\percent}}".format(
	        fixedMissID * 100))
	if legend is True:
		for m in models2compare:
			plots[0, 0].plot(1, 1, label=m.label)
		plots[0, 0].setXlim((-0.5, 0.5))
		plots[0, 0].axes.set_axis_off()
		plots[0, 0].legend(loc='best')
		plots[0, 1].axes.set_axis_off()

	varMap = {'p': 0, 'cosTheta': 1, 'phi': 2}
	plotVars = tuple(varMap[i] for i in plotVars)

	for iRate, (species, hypothesis, rateLabel,
	            rateRange) in enumerate(combinations):
		for iVar, _ in enumerate(binEdges):
			if iVar not in plotVars:
				continue
			plot = plots[plotVars.index(iVar) + legendOffset, iRate]

			summedAxes = tuple(i for i in range(3) if i != iVar)
			for m in models2compare:
				label = m.label
				rate, rateUnc = _calcRateAndRateUncertainty(m, species, hypothesis, summedAxes)
				if reference4Ratio is None:
					plot.plotErrorBar(0.5 * (binEdges[iVar][1:] + binEdges[iVar][:-1]), # pylint: disable=unnecessary-list-index-lookup
					    rate * 100,
						xerr=0.5 * (binEdges[iVar][1:] - binEdges[iVar][:-1]), # pylint: disable=unnecessary-list-index-lookup
					    yerr=rateUnc * 100,
					    label=label)
				else: # plot ratio w/r/t reference
					rateRef, _ = _calcRateAndRateUncertainty(models2compare[reference4Ratio], species, hypothesis, summedAxes)
					ratio = math.divide(rate, rateRef, where=rateRef!=0, whereNotValue=np.nan)
					plot.plot(0.5 * (binEdges[iVar][1:] + binEdges[iVar][:-1]),
					          ratio,
					    marker = 'o', linestyle = '',
					          label=label)


			plot.setXYgrid()
			plot.setXticks(10)
			plot.setYticks(6)
			plot.setXlabel(binningVariableLabels[iVar])
			if reference4Ratio is None:
				plot.setYlim(rateRange)
				plot.setYlabel(rateLabel + r'[\si\percent]')
			else:
				plot.setYlim(ylimit4Ratio if ylimit4Ratio is not None else (0.0, 2.0))
				plot.setYlabel(f'{rateLabel} Ratio')
	if legend == 'in':
		plots[0,1].legend(loc='best', fontsize=style.legendFontSize * 0.8)
	if datasetLabel is not None:
		plots[0,1].addTitleRight(datasetLabel)
	plots.finish()
	return plots


def plotROC(models2compare,
            xlim,
            style,
            syscorrfw_roc=None,
            species='K',
            missidSpecies='pi',
            ylim=None,
            legend=True,
            figHeight=5,
   datasetLabel=None):
	'''Plot ROC curve (efficiency vs misidentification rate) for one or multiple models

	:param models2compare: List of models to compare
	:type models2compare: list
	:param xlim: Set limits of x-axis
	:type xlim: tuple of float
	:param style: Object containing plotting methods, setting style of the plot
	:param syscorrfw_roc: If given, plots reference ROC curve from systematic corrections framework, defaults to None
	:type syscorrfw_roc: dict or None, optional
	:param species: The particle species, defaults to 'K'
	:type species: str, optional
	:param missidSpecies: The particle species that is missidentified, defaults to 'pi'
	:type missidSpecies: str, optional
	:param ylim: Set limits of y-axis, defaults to None
	:type ylim: tuple of float, optional
	:param legend: Boolean which determines whether a legend is plotted, defaults to True
	:type legend: bool, optional
	:param figHeight: The figure height, defaults to 5
	:type figHeight: int, optional
	:param datasetLabel: Title that is added to the plot, defaults to None
	:type datasetLabel: str or None, optional
	'''
	plot = style.getPlot1D(figWidth=6, figHeight=figHeight)
	for m in models2compare:
		label = "{1:.4f}  {0}".format(m.label,
		                              m.aurocs[species][missidSpecies])
		plot.plot(m.missids[missidSpecies][species][1:-1],
		          m.efficiencies[species][1:-1],
		          label=label)
	if syscorrfw_roc:
		plot.plot(syscorrfw_roc['efficiency_fake_rate'][1],
		          syscorrfw_roc['efficiency_fake_rate'][0],
		          'x',
		          label='SysCorrFw',
		          color=style.colorScheme.darkGray,
		          zorder=1.2)
	plot.setXlim(xlim)
	if ylim is None:
		plot.setYlim((0.5, 1.0))
	else:
		plot.setYlim(ylim)
	plot.setXticks(10)
	plot.setXshowMinorTicks(2)
	plot.setYticks(5)
	plot.setXYgrid()
	plot.setXlabel(r"${0}$ miss-ID Rate".format(
	    missidSpecies.replace('pi', r'\pi')))
	plot.setYlabel(r"${0}$ Efficiency".format(species.replace('pi', r'\pi')))
	if legend:
		plot.legend(loc='best' if legend is True else legend)
	plot.finish()
	if datasetLabel is not None:
		plot.addTitleRight(datasetLabel)
	return plot


def plotROCRatios(models2compare,
                  refModel,
                  xlim,
                  style,
                  syscorrfw_roc=None,
                  species='K',
                  missidSpecies='pi',
                  ylim=None,
                  legend=True,
                  figHeight=5,
                  datasetLabel=None):
	'''Plot ratios of ROC curves (efficiency ratio vs misidentification rate), comparing the models in 'models2compare' to a reference model 'refModel'.

	:param models2compare: List of models to compare
	:type models2compare: list
	:param refModel: Reference model
	:param xlim: Set limits of x-axis
	:type xlim: tuple of float
	:param style: Object containing plotting methods, setting style of the plot
	:param syscorrfw_roc: If given, plots reference ROC curve from systematic corrections framework, defaults to None
	:type syscorrfw_roc: dict or None, optional
	:param species: The particle species, defaults to 'K'
	:type species: str, optional
	:param missidSpecies: The particle species that is missidentified, defaults to 'pi'
	:type missidSpecies: str, optional
	:param ylim: Set limits of y-axis, defaults to None
	:type ylim: tuple of float, optional
	:param legend: Boolean which determines whether a legend is plotted, defaults to True
	:type legend: bool, optional
	:param figHeight: The figure height, defaults to 5
	:type figHeight: int, optional
	:param datasetLabel: Title that is added to the plot, defaults to None
	:type datasetLabel: str or None, optional
	'''
	plot = style.getPlot1D(figWidth=6, figHeight=figHeight)
	missIDsRef = np.array(refModel.missids[missidSpecies][species])
	efficienciesRef = np.array(refModel.efficiencies[species])
	for m in models2compare:
		label = "{1:.4f}  {0}".format(m.label,
		                              m.aurocs[species][missidSpecies])
		missIDs = np.array(m.missids[missidSpecies][species])
		efficiencies = np.array(m.efficiencies[species])
		interpolatedAtRef = math.divide(scipy.interpolate.interp1d(
		    missIDs, efficiencies, bounds_error=False)(missIDsRef),
		                                efficienciesRef,
		                                where=(efficienciesRef != 0.),
		                                whereNotValue=np.nan)
		plot.plot(missIDsRef, interpolatedAtRef, label=label)
	if syscorrfw_roc:
		plot.plot(syscorrfw_roc['efficiency_fake_rate'][1],
		          syscorrfw_roc['efficiency_fake_rate'][0],
		          'x',
		          label='SysCorrFw',
		          color=style.colorScheme.darkGray,
		          zorder=1.2)
	plot.setXlim(xlim)
	if ylim is None:
		plot.setYlim((0.6, 1.4))
	else:
		plot.setYlim(ylim)
	plot.setXticks(10)
	plot.setXshowMinorTicks(2)
	plot.setYticks(10)
	plot.setXYgrid()
	plot.setXlabel(r"${0}$ miss-ID Rate".format(
	    missidSpecies.replace('pi', r'\pi')))
	plot.setYlabel(r"${0}$ Efficiency Ratio".format(
	    species.replace('pi', r'\pi')))
	if datasetLabel is not None:
		plot.addTitleRight(datasetLabel)
	if legend:
		plot.legend(loc='best' if legend is True else legend)
	plot.finish()
	return plot


def plotPdistributionMom(model, style):
	'''Plot the 2D correlation of the momentum :math:`(|p|)` and the probability ratio :math:`\\frac{P(K)}{\\sum_S P(S)}`

	:param model: Model to evaluate
	:param style: Object containing plotting methods, setting style of the plot
	'''
	plots = style.getSubplots2D(ncols=2, nsubplots=len(model.classMap))

	for i, p in enumerate(model.classMap):
		plot = plots.getItemRowMajor(i)
		counts, _, _, _ = plot.hist2d(
		    model.datasetTestAuxNp[model.datasetTestLabelsNp ==
		                           model.classMap[p], 0],
		    model.probRatiosNp[model.datasetTestLabelsNp == model.classMap[p],
		                       model.classMap['K']],
		    weights=model.datasetTestAuxNp[model.datasetTestLabelsNp ==
		                                   model.classMap[p], -1],
		    range=((0, 6), (0, 1)))
		plot.setXYgrid()
		plot.setXlabel(r'$\absVecp$ [\si\GeVc]')
		plot.setYlabel(r'$P(K) / \sum_\mathrm{S} P(S)$')
		plot.addTitleRight('${0}$'.format(p.replace('pi', r'\pi')))
		plot.setXticks(6)
		plot.setZlog()
		plot.setZlim((1, np.nanmax(counts)))
		plot.setZshowColorBar()
	plots[0, 0].addTitleLeft(model.label)
	plots.finish()
	return plots


def plotPdistributioncosT(model, style):
	'''Plot the 2D distribution of :math:`\\cos(\\theta)` vs the probability ratio :math:`\\frac{P(K)}{\\sum_S P(S)}`

	:param model: Model to evaluate
	:param style: Object containing plotting methods, setting style of the plot
	'''
	plots = style.getSubplots2D(ncols=2, nsubplots=len(model.classMap))

	for i, p in enumerate(model.classMap):
		plot = plots.getItemRowMajor(i)
		counts, _, _, _ = plot.hist2d(
		    model.datasetTestAuxNp[model.datasetTestLabelsNp ==
		                           model.classMap[p], 1],
		    model.probRatiosNp[model.datasetTestLabelsNp == model.classMap[p],
		                       model.classMap['K']],
		    weights=model.datasetTestAuxNp[model.datasetTestLabelsNp ==
		                                   model.classMap[p], -1],
		    range=((-1, 1), (0, 1)))
		plot.setXYgrid()
		plot.setXlabel(r'$\cos\theta$')
		plot.setYlabel(r'$P(K) / \sum_\mathrm{S} P(S)$')
		plot.addTitleRight('${0}$'.format(p.replace('pi', r'\pi')))
		plot.setXticks(6)
		plot.setZlog()
		plot.setZlim((1, np.nanmax(counts)))
		plot.setZshowColorBar()
	plots[0, 0].addTitleLeft(model.label)
	plots.finish()
	return plots


def plotPdistribution(model, style):
	'''Plot both the correlation of :math:`\\cos(\\theta)` vs the probability ratio :math:`\\frac{P(K)}{\\sum_S P(S)}` and the
	correlation of the momentum :math:`(|p|)` vs the probability ratio.

	:param model: Model to evaluate
	:param style: Object containing plotting methods, setting style of the plot
	'''
	plotPdistributionMom(model, style)
	plotPdistributioncosT(model, style)


def plot2DPerformance(fixedMissID,
                      missIDyMax,
                      models2compare,
                      style,
                      binEdges,
                      binningVariableLabels,
                      combinations='K_pi',
       datasetLabel=None):
	'''Creates 2D plot of efficiency and misidentification rate for one or multiple models.

	:param fixedMissID: Threshold of the misidentification rate
	:type fixedMissID: float
	:param missIDyMax: Maximum value for misidentification rate
	:type missIDyMax: float
	:param models2compare: List of models to compare
	:type models2compare: list
	:param style: Object containing plotting methods, setting style of the plot
	:param binEdges: The bin edges for the performance histograms
	:param binningVariableLabels: The labels for the variables
	:type binningVariableLabels: list of str
	:param combination: Combination of particles to plot, currently only 'K_pi' and 'pi_K' implemented
	:type combination: str, optional
	:param datasetLabel: Title that is added to the plot, defaults to None
	:type datasetLabel: str or None, optional
	'''

	if combinations == 'K_pi':
		combinations = [
		    ('K', 'K', r'$K$ Efficiency [\si\percent]', (0.0, 105)),
		    ('pi', 'K', r'$\pi$ Miss-ID [\si\percent]', (0.0, missIDyMax)),
		]
	elif combinations == 'pi_K':
		combinations = [
		    ('pi', 'pi', r'$\pi$ Efficiency [\si\percent]', (0.0, 105)),
		    ('K', 'pi', r'$K$ Miss-ID [\si\percent]', (0.0, missIDyMax)),
		]

	else:
		raise Exception(f"Unknown combination '{combinations}'")

	# plots = style.getSubplots1D(nrows=len(binEdges)+(1 if plotPhi else 0), ncols=len(combinations), figWidth=8, figHeight=8 if plotPhi else 6)
	# plots[0,0].addTitleLeft(r"Fixed average miss-ID rate: \SI{{{0:.1f}}}{{\percent}}".format(fixedMissID*100))
	for m in models2compare:
		plots = style.getSubplots2D(1, 2)
		for iRate, (species, hypothesis, rateLabel,
		            rateRange) in enumerate(combinations):
			calcEffMissidAndPerformanceHists(m, binEdges,
			              fixedMissID=fixedMissID,
			              missIDSpecies=(combinations[1][0],
			                             combinations[1][1]))
			# plot = style.getPlot2D()
			plot = plots.getItemRowMajor(iRate)
			plot.addTitleLeft(m.label, fontsize=style.legendFontSize * 0.7)
			plot.addTitleRight(
			    r"{0}miss-ID rate: \SI{{{1:.1f}}}{{\percent}}".format(
			  "" if datasetLabel is None else f'{datasetLabel}: ',
			        fixedMissID * 100),
			    fontsize=style.legendFontSize * 0.7)

			summedAxes = (2, )
			total = np.sum(m.histsSpecies[species], axis=summedAxes)
			identified = np.sum(
			    m.histsSpeciesAsHypothesis[species][hypothesis],
			    axis=summedAxes)
			# totalUncSquared = np.sum(m.histsSpeciesSquared[species],
			#                          axis=summedAxes)
			# identifiedUncSquared = np.sum(
			#     m.histsSpeciesAsHypothesisSquared[species][hypothesis],
			#     axis=summedAxes)
			rate = np.empty_like(total)
			np.divide(identified, total, out=rate, where=total > 0)
			# rateUnc = calcEfficiencyUncertainty(identified, total,
			#                                     identifiedUncSquared,
			#                                     totalUncSquared)
			X, Y = np.meshgrid(binEdges[0], binEdges[1])
			plot.axes.pcolormesh(X, Y, rate.T * 100)

			# plot.setXYgrid()
			plot.setXticks(10)
			plot.setYticks(10)
			plot.setXlabel(binningVariableLabels[0])
			plot.setYlabel(binningVariableLabels[1])
			plot.setZshowColorBar()
			plot.setZlabel(rateLabel)
			plot.setZlim(rateRange)
		plots.finish()
		# return
