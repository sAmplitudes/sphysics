# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Different metrics for the pytorch training, Created on Wednesday 01 06 2022
'''

from __future__ import absolute_import, print_function, division, annotations

import torchmetrics
import torch


class Loss:
	'''Return average loss over number of batches.
	'''
	name = "loss"
	def __init__(self):
		self._loss = 0.
		self._nBatches = 0
	def __call__(self, loss, **kwargs):
		self._nBatches+=1
		self._loss += loss
	def compute(self):
		return self._loss/self._nBatches

class Accuracy:
	'''Return Accuracy
	'''
	name = "accuracy"
	def __init__(self):
		self._correct = 0
		self._nEvents = 0
	def __call__(self, labels, predictedBest, **kwargs):
		self._correct += (predictedBest == labels).type(torch.float).sum().item()
		self._nEvents += len(labels)
	def compute(self):
		return self._correct/self._nEvents

class AUROC:
	'''Return Area Under the Receiver Operating Characteristic curve (AUROC).
	'''
	name = "AUROC"
	def __init__(self):
		self._metric = torchmetrics.AUROC('multiclass', num_classes=2)
	def __call__(self, labels, predictedLogProb, **kwargs):
		return self._metric(predictedLogProb, labels) # pylint: disable=not-callable
	def compute(self):
		return float(self._metric.compute().cpu().numpy())

class IDProb:
	def __init__(self, species, speciesHypothesis, likelyhoodRatioThreshold):
		self._species = species
		self._speciesHypothesis = speciesHypothesis
		self._likelihoodRatioThreshold = likelyhoodRatioThreshold
		self._nEvents = 0
		self._correct = 0
	def __call__(self, labels, predictedProb, **kwargs):
		isSpecies = labels == self._species
		likelihoodRatio = predictedProb[isSpecies,self._speciesHypothesis]/torch.sum(predictedProb[isSpecies], 1)
		nEvents = isSpecies.type(torch.float).sum().item()
		correct = (likelihoodRatio > self._likelihoodRatioThreshold).type(torch.float).sum().item()
		self._nEvents += nEvents
		self._correct += correct
		return correct/nEvents
	def compute(self):
		return self._correct/self._nEvents

class Efficiency(IDProb):
	'''Return efficiency (true positive rate) for a given `species`.

    :param species: The true class
    :type species: int
    :param likelyhoodRatioThreshold: Threshold of the likelihood ratio
    :type likelyhoodRatioThreshold: float
    '''
	def __init__(self, species, likelyhoodRatioThreshold):
		super().__init__(species, species, likelyhoodRatioThreshold)
		self.name = f"{species}_{likelyhoodRatioThreshold:.2f}"

class MissID(IDProb):
	'''Return misidentification rate

	:param species: True class
	:type species: int
	:param speciesHypothesis: Hypothesized class
	:type speciesHypothesis: int
	:param likelyhoodRatioThreshold: Threshold of likelihood ratio
	:type likelyhoodRatioThreshold: float
	'''
	def __init__(self, species, speciesHypothesis, likelyhoodRatioThreshold):
		super().__init__(species, speciesHypothesis, likelyhoodRatioThreshold)
		self.name = f"{species}-{speciesHypothesis}_{likelyhoodRatioThreshold:.2f}"
