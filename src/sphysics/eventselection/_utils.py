# coding: utf-8
'''
:Author: Stefan Wallner

Description: Created on Friday 19 05 2023
'''

from __future__ import absolute_import, print_function, division, annotations

from typing import Callable, Sequence, Union

import pandas as pd
import numpy as np
import tabulate

from ._variables import Variables
from .hist import SparseHist
from .. import utils
from .. import math

log = utils.Logger("eventselection")



def addLuminosityWeight(sample: Variables, channelLuminosity: dict,
   targetLuminosity: float, weightVariableName: str='luminosity_weight') -> Variables:
	"""Calculates event weights to reweight subsamples with different luminosities to a common target luminosity

	Args:
		sample (Variables): Sample whose subsamples, i.e. channels should be weighted
		channelLuminosity (dict): Dictionary with channel name as key and luminosity as value
		targetLuminosity (float): Target luminosity.
		weightvAriableName (str, optional): Name of the variable in which the weights are store. Defaults to 'weights'.

	Returns:
		Variables: Modified sample with weights
	"""
	sample.addVariable(weightVariableName)
	sample[weightVariableName] = np.full(sample.nEvents, np.nan)
	for channel in channelLuminosity:
		if channel not in sample.getChannels():
			log.raiseException(Exception, f'Channel {channel} not in sample!')
		if not np.isnan(channelLuminosity[channel]):
			sample[weightVariableName][sample.idx(channel)] = targetLuminosity / channelLuminosity[channel]
	if np.any(np.isnan(sample[weightVariableName])):
		log.raiseException(Exception, 'Weight could not be set for all entries!')
	return sample




def getWeightedNeventsTable(sample: Variables, weight: np.ndarray|str = None) -> tuple[pd.DataFrame, pd.io.formats.Styler]:
	"""Get nicely formatted table of number of events and event fractions taking into account event weights.

	Args:
		sample (Variables): Sample object
		weight (str | np.ndarray, optional): Name of weight member variable or array of weights. Defaults to 'weight'.

	Returns:
		str: String containing the table
		tuple[pd.DataFrame, pd.io.formats.Styler]: dataframe with number of events and nicely formatted version of the dataframe
	"""
	nEvents = sample.getWeightedNevents(weight)
	ratios = {c: nEvents[c]/nEvents['total'] for c in nEvents}
	rows = []
	for channel in sorted(sample.getChannels(), key=lambda c: ratios[c], reverse=True):
		rows.append([channel, int(nEvents[channel]), ratios[channel]*100])
	df = pd.DataFrame(rows, columns=['Channel', '# events', '# fraction [%]'])
	df = df.set_index('Channel')
	return df, df.style.format(precision=2, thousands=',')


def getWeightedEfficiencyTable(mask: np.ndarray, fullSample: Variables, weight: str = None) -> tuple[pd.DataFrame, pd.io.formats.Styler]:
	"""Get nicely formatted table of events, efficiencies and event factions in a sample selected according to `mask` from `fullSample`.

	Args:
		mask (np.ndarray): Selection mask of entries in `fullSample`
		fullSample (Variables): Sample with all events
		weight (str, optional): Name of variable containing weights. Defaults to 'weight'.

	Returns:
		tuple[pd.DataFrame, pd.io.formats.Styler]: dataframe with data nicely formatted version of the dataframe
	"""
	weight = weight if weight is not None else 'weight'

	nEventsFull = fullSample.getWeightedNevents(weight)
	nEventsFull2 = fullSample.getWeightedNevents(fullSample['weight' if weight is None else weight]**2)
	nEventsSelected = {c: np.sum(fullSample[weight][fullSample.idx(c)][mask[fullSample.idx(c)]]) for c in fullSample.getChannels()}
	nEventsSelected2 = {c: np.sum(fullSample[weight][fullSample.idx(c)][mask[fullSample.idx(c)]]**2) for c in fullSample.getChannels()}
	ratios = {c: math.divide(nEventsSelected[c],nEventsSelected['total'], where = nEventsSelected['total']>0, whereNotValue=np.nan)  for c in nEventsSelected}
	ratioUncertainty= {c: ratios[c]*math.sqrt( (1-2*math.divide(nEventsSelected[c],nEventsSelected['total'], where=nEventsSelected['total']>0, whereNotValue=np.nan)) \
													          * math.divide(nEventsSelected2[c], nEventsSelected[c]**2, where=nEventsSelected[c]>0, whereNotValue=np.nan) \
														 + math.divide(nEventsSelected2['total'],nEventsSelected['total']**2, where=nEventsSelected['total']>0, whereNotValue=np.nan) ) for c in nEventsSelected}
	efficiency = {c: math.divide(nEventsSelected[c],nEventsFull[c], where=nEventsFull[c]>0, whereNotValue=np.nan) for c in nEventsSelected}
	efficiencyUncertainty = {c: efficiency[c]*math.sqrt( (1-2*math.divide(nEventsSelected[c],nEventsFull[c], where=nEventsFull[c]>0, whereNotValue=np.nan)) \
													          * math.divide(nEventsSelected2[c], nEventsSelected[c]**2, where=nEventsSelected[c]>0, whereNotValue=np.nan) \
														 + math.divide(nEventsFull2[c],nEventsFull[c]**2, where=nEventsFull[c]>0, whereNotValue=np.nan) ) for c in nEventsSelected}
	rows = []
	for channel in sorted(fullSample.getChannels(), key=lambda c: ratios[c], reverse=True):
		rows.append([channel, efficiency[channel]*100, efficiencyUncertainty[channel]*200, int(nEventsSelected[channel]), ratios[channel]*100, ratioUncertainty[channel]*100])
	df = pd.DataFrame(rows, columns=['Channel', 'efficiency [%]', 'unc. efficiency [%]', '# events', '# fraction [%]', 'unc. # fraction [%]'])
	df = df.set_index('Channel')
	return df, df.style.format(precision=2, thousands=',')


def thresholdScan(
    sample: Variables, cutFunction: Callable, thresholds: int|np.ndarray = 50,
    weight: str = None,
	signalChannelName: str = 'signal',
	totalChannelName: str = 'total',
	efficiencyScaling: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str,np.ndarray], np.ndarray, dict[str,np.ndarray], np.ndarray, dict[str,np.ndarray], np.ndarray, dict[str,np.ndarray]]:
	"""Perform a threshold scan using weights per default

	Args:
		sample (Variables): Data sample
		cutFunction (function): Cut function with signature (sample, threshold) -> cutMask, which determine the cut mask for a given threshold
		thresholds (int|np.ndarray,np.ndarray): List of thresholds to test (in increasing order) or number of thresholds points.
			If it is a number of threshold points, a heuristik tries to find the threshold to cover the
			ROC curve as good as possible. The actual number of thresholds can be slightly different from
			the given number of thresholds due to rounding effects.

		weight (str, optional): Name of variable containing weights. Defaults to 'weight'.
		signalChannelName (str, optional): Name of channel that define the signal
		totalChannelName (str, optional): Name of channel that has all events
		efficiencyScaling (float, optional): Scale all efficiencies by this factor, i.e. efficiencies and mis-id rates. Default no scaling.

	Returns:
		   - list of thresholds
		   - list of efficiences
		   - dict of lists of impurities for each channel
		   - list of purities, i.e. 1-sum(impurities)
		   - list of miss-id rates for the channels, i.e. efficiency of the non-signal channels
		   - list of efficiency uncertainties
		   - dict of lists of impurity uncertainties for each channel
		   - list of purity uncertainties
		   - list of miss-id rate uncertainties
	"""

	if isinstance(thresholds, int):
		# find best scattering of thresholds by first performing a rough thresholdScan
		scanThresholds = np.hstack([np.linspace(0,0.1,5), np.linspace(0.1, 0.9, 9)[1:-1], np.linspace(0.9,1.0,5)])
		scanThresholds, eff, _, _, missid, *_ = thresholdScan(sample, cutFunction, scanThresholds,
														weight=weight,signalChannelName=signalChannelName, totalChannelName=totalChannelName)
		thresholdRangeWeight = np.zeros_like(scanThresholds[1:])
		for rate in list(missid.values())+[eff]:
			thresholdRangeWeight += 2*math.divide(rate[1:] - rate[:-1], rate[1:] + rate[:-1], where=(rate[1:] + rate[:-1])>0, whereNotValue=0.0)
		thresholdRangeWeight /= np.sum(thresholdRangeWeight)
		nAdditionalThresholds = max(0, thresholds-scanThresholds.size)
		newThresholds = []
		for i, threshold in enumerate(scanThresholds[:-1]):
			n = max(0, int(np.round(nAdditionalThresholds*thresholdRangeWeight[i])))
			newThresholds += list(np.linspace(threshold, scanThresholds[i+1], 2+n)[:-1])
		thresholds = np.array(newThresholds)


	efficiencies = []
	efficiencyUnc = []
	purities = []
	purityUnc = []
	impurities = {
	    imp: []
	    for imp in sample.getChannels() if imp not in (totalChannelName, signalChannelName)
	}
	impurityUnc = {
	    imp: []
	    for imp in sample.getChannels() if imp not in (totalChannelName, signalChannelName)
	}
	missidrates = {
	    imp: []
	    for imp in sample.getChannels() if imp not in (totalChannelName, signalChannelName)
	}
	missidrateUnc = {
	    imp: []
	    for imp in sample.getChannels() if imp not in (totalChannelName, signalChannelName)
	}
	for threshold in thresholds:
		mask = cutFunction(sample, threshold)
		df, _ = getWeightedEfficiencyTable(mask, sample, weight=weight)
		efficiencies.append(df['efficiency [%]'][signalChannelName])  # pylint: disable=unsubscriptable-object
		efficiencyUnc.append(df['unc. efficiency [%]'][signalChannelName])  # pylint: disable=unsubscriptable-object
		for imp in impurities:
			impurities[imp].append(df['# fraction [%]'][imp])  # pylint: disable=unsubscriptable-object
			impurityUnc[imp].append(df['unc. # fraction [%]'][imp])  # pylint: disable=unsubscriptable-object
		for c in missidrates:
			missidrates[c].append(df['efficiency [%]'][c])  # pylint: disable=unsubscriptable-object
			missidrateUnc[c].append(df['unc. efficiency [%]'][c])  # pylint: disable=unsubscriptable-object
		purities.append(df['# fraction [%]'][signalChannelName])  # pylint: disable=unsubscriptable-object
		purityUnc.append(df['unc. # fraction [%]'][signalChannelName])  # pylint: disable=unsubscriptable-object
	thresholds = np.array(thresholds)
	purities = np.array(purities)
	purityUnc = np.array(purityUnc)
	efficiencies = np.array(efficiencies)*efficiencyScaling
	efficiencyUnc = np.array(efficiencyUnc)*efficiencyScaling
	impurities = {k: np.array(v) for k,v in impurities.items()}
	impurityUnc = {k: np.array(v) for k,v in impurityUnc.items()}
	missidrates = {k: np.array(v)*efficiencyScaling for k,v in missidrates.items()}
	missidrateUnc = {k: np.array(v)*efficiencyScaling for k,v in missidrateUnc.items()}
	return thresholds, efficiencies, impurities, purities, missidrates, efficiencyUnc, impurityUnc, purityUnc, missidrateUnc


def pidHist(distributions: Union[Sequence[np.ndarray],np.ndarray], pid2Name: dict=None, headers: Sequence[str]=None, sortByCount=False,
            weights: Union[Sequence[np.ndarray],np.ndarray] = None):
	"""Two tables, one for the absolute one one for the relative amount counts of each id, are printed.

	Args:
		distributions (Union[Sequence[np.ndarray],np.ndarray]): List if input arrays with ids
		pid2Name (dict, optional): Dictionary, that mapes the ids to names for printing of the tables. Defaults to None.
		headers (Sequence[str], optional): Headline of table column, one for each input array. Defaults to None.
		sortyByCount (bool, optional): Sort rows by count of hist distribution not by id. Defaults to False.
		weights (Union[Sequence[np.ndarray],np.ndarray], optional): Weight for entries, i.e. each entry counts for w instead of 1. Defaults to None.
	"""
	if not isinstance(distributions, list):
		distributions = [distributions]
	if not isinstance(weights, list):
		weights = [weights]*len(distributions)
	if len(distributions) != len(weights):
		log.raiseException(Exception, "distributions and weights have different number of entries!")
	table = []
	tablePercent = []
	hists = []
	distributionTotals = [distribution.size if weight is None else math.sum(weight) for distribution, weight in zip(distributions, weights)]
	if distributions[0].size > 0:
		binIds = set()
		for distribution, weight in zip(distributions, weights):
			hist = SparseHist(1.0, -0.5)
			hist.fill(distribution, weights=weight)
			hists.append(hist)
			binIds = binIds.union(list(hist.counts.keys()))
		if not sortByCount:
			binIds = sorted(list(binIds), key=abs)
			binIds = sorted(list(binIds), key=np.sign)
		else:
			binIds = sorted(binIds, key=lambda i: hists[0].counts[i], reverse=True)
		for binId in binIds:
			table.append([int(binId)])
			tablePercent.append([int(binId)])
			for distribution, distributionTotal, hist in zip(distributions, distributionTotals, hists):
				table[-1].append(hist.counts[binId] if binId in hist.counts else 0)
				tablePercent[-1].append(hist.counts[binId]/distributionTotal*100 if binId in hist.counts else 0.)
		table.append(["{0:>8s}".format("NaN")])
		tablePercent.append(["{0:>8s}".format("NaN")])
		for distribution, distributionTotal in zip(distributions, distributionTotals):
			table[-1].append(np.sum(np.isnan(distribution)))
			tablePercent[-1].append(np.sum(np.isnan(distribution))/distributionTotal*100)
		table.append(["{0:>8s}".format("total")])
		tablePercent.append(["{0:>8s}".format("total")])
		for distributionTotal in distributionTotals:
			table[-1].append(distributionTotal)
			tablePercent[-1].append(100.)
		for row in table:
			if isinstance(row[0], int):
				if pid2Name is not None and row[0] in pid2Name:
					row[0] = pid2Name[row[0]]
				else:
					row[0] = "{0:8d}".format(row[0])
		for row in tablePercent:
			if isinstance(row[0], int):
				if pid2Name is not None and row[0] in pid2Name:
					row[0] = pid2Name[row[0]]
				else:
					row[0] = "{0:8d}".format(row[0])
		if headers is not None:
			headers = ['Ids'] + list(headers)
		else:
			headers = ()
		print(tabulate.tabulate(table, floatfmt=',.0f', stralign='right', headers=headers))
		print(tabulate.tabulate(tablePercent, floatfmt='.2f', stralign='right', headers=headers))
