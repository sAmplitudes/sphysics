# coding: utf-8
'''
Created on Tuesday 07 06 2022
Author: Stefan Wallner
Description: Special functions for Belle2 hadron ID
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
import torchinfo
import torch
from torch import nn
from torch.utils.data import TensorDataset

from ...eventselection import VariablesBase

from .._training import TrainingHistory


def variables2Dataset(classId: int, variables: VariablesBase, inputVariables: list[str], auxVariables: list[str], device: torch.device = None) -> TensorDataset:
	"""Extracts the `TensorDataset` for training and evaluation from `variables`

	Args:
		classId (int): Id of the class of the particles in `variables` (must all be the same!)
		variables (VariablesBase): Data
		inputVariables (list[str]): List of variables used for training
		auxVariables (list[str]): List of auxiliary variable stored in the dataset
		classMap (dict[str,int]): Mapping from species name (`K`, `pi`, ...) to class indices in the classifier
		device (torch.device, optional): Device on which the tensors are initialized. Defaults to None.

	Returns:
		TensorDataset: Dataset to be used for taining
	"""
	dataInput  = np.empty((0, len(inputVariables)))
	dataLabels = np.empty((0, ), dtype=int)
	dataAux	= np.empty((0, len(auxVariables)))



	dataInputSpecies = np.empty((variables.nEvents, dataInput.shape[1]))
	for i, inputVariable in enumerate(inputVariables):
		dataInputSpecies[:,i] = variables[inputVariable]
		if not inputVariable.startswith('pid'): # check for nans
			if np.any(np.isnan(dataInputSpecies[:,i])):
				raise Exception(inputVariable)
	dataAuxSpecies = np.empty((variables.nEvents, dataAux.shape[1]))
	for i, auxVariable in enumerate(auxVariables):
		dataAuxSpecies[:,i] = variables[auxVariable]

	dataInput = np.concatenate((dataInput, dataInputSpecies))
	dataAux = np.concatenate((dataAux, dataAuxSpecies))
	dataLabels = np.concatenate((dataLabels, np.full(dataInputSpecies.shape[0], classId, dtype=int)))



	dataInput = torch.tensor( dataInput, dtype=torch.float32, device=device)
	dataLabels = torch.tensor( dataLabels, dtype=torch.int64, device=device)
	dataAux = torch.tensor( dataAux, dtype=torch.float32, device=device)
	datasetFull = TensorDataset(dataInput, dataLabels, dataAux)

	return datasetFull


def multipleVariables2Dataset(classIds: list[int], multipleVariables: list[VariablesBase],
                              inputVariables: list[str], auxVariables: list[str], device: torch.device = None) -> TensorDataset:
	"""Extracts the `TensorDataset` for training and evaluation from `variables`

	Args:
		classIds (list[int]): Id of the class of the particles, one for each variables in multipleVariables
		multipleVariables (list[VariablesBase]): List of datasets
		inputVariables (list[str]): List of variables used for training
		auxVariables (list[str]): List of auxiliary variable stored in the dataset
		classMap (dict[str,int]): Mapping from species name (`K`, `pi`, ...) to class indices in the classifier
		device (torch.device, optional): Device on which the tensors are initialized. Defaults to None.

	Returns:
		TensorDataset: Dataset to be used for taining
	"""

	datasetFull = None
	for classId, variables in zip(classIds, multipleVariables):
		dataset = variables2Dataset(classId, variables, inputVariables, auxVariables, device)
		if datasetFull is None:
			datasetFull = dataset
		else:
			newTensors = []
			for tFull, tSub in zip(datasetFull.tensors, dataset.tensors):
				newTensors.append(torch.cat((tFull, tSub)))
			datasetFull = torch.utils.data.TensorDataset(*newTensors)
	return datasetFull



class NeuralNetwork(nn.Module):
	def __init__(self, inputVariables, auxVariables, classMap, projectTag, means, stds, listCutVariables=None):
		"""Documentation:
			listCutVariables (list): List of variables that will be changed and assigned another value
			listCutVariables= [
			[varToCut,[(varReference,startrange,endrange,setval)]],
			]
		varToCut(str) : Name of the variable that you want to change
		setval (float) : value of the variable that will be set
		varReference (str) : variable used as reference to define the range in which varToCut will be set
		startrange/endrange (float / None): values that define the interval in varReference in which varToCut will be set to setval

		Setting varToCut to setval is done BEFORE standardization and BEFORE handling NaNs.

		Ex:
			listCutVariables= [
			['pidLogLikelihood_Of_e_From_SVD',[('p',2,3,5)]],
			]
		The variables pidLogLikelihood_Of_e_From_SVD will take the value 5 in the events where the momentum is between 2 and 3
		"""
		super(NeuralNetwork, self).__init__()
		self.means = means
		self.stds = stds
		self.inputVariables = inputVariables
		self.auxVariables = auxVariables
		self.classMap = classMap
		self.projectTag = projectTag
		self.history = TrainingHistory()
		self.iInputLogLikelihoods = [i for i, inputVariable in enumerate(inputVariables) if 'pidLogLikelihood' in inputVariable]
		self.iInputLikeRatios	 = [i for i, inputVariable in enumerate(inputVariables) if 'pidLikelihoodRatio' in inputVariable]

		self.nn = nn.Sequential()
		self.normLayer = nn.LogSoftmax(1)

		self.listCutVariables=listCutVariables if listCutVariables is not None else []

	def prepareInput(self, x):
###########################################################################################
# Update: 2022-12-12 : Introduction to the possibility to cut via self.listCutVariables
# For old models the variable was not implemented so it will give problems with loop. Hence:
		if not hasattr(self, 'listCutVariables'):
			self.listCutVariables=[]
###########################################################################################
		x = x[:] # need to copy x as it will be modified below
		for cutVariables in self.listCutVariables:
			varToCut=cutVariables[0]
			indexVarCut=self.inputVariables.index(varToCut)
			varReference=cutVariables[1][0][0]
			indexVarRef=self.inputVariables.index(varReference)
			startrange=cutVariables[1][0][1]
			endrange=cutVariables[1][0][2]
			setval=cutVariables[1][0][3]
			mask = None
			if (startrange is not None) & (endrange is not None):
				mask=(x[:,indexVarRef]>=startrange) & (x[:,indexVarRef]<=endrange)
			elif (startrange is not None) & (endrange is None):
				mask=(x[:,indexVarRef]>=startrange)
			elif (startrange is None) & (endrange is not None):
				mask=(x[:,indexVarRef]<=endrange)
			if mask is not None:
				x[mask,indexVarCut]=setval
			else:
				x[:,indexVarCut]=setval

		xNormed = (x-self.means)/self.stds
		for i in self.iInputLogLikelihoods:
			xNormed[torch.isnan(xNormed[:,i]),i] = 1.0
		for i in self.iInputLikeRatios:
			xNormed[torch.isnan(xNormed[:,i]),i] = -1.0
		return xNormed

	def forward(self, x):
		xNormed = self.prepareInput(x)
		logits = self.nn(xNormed)
		logProb = self.normLayer(logits)
		return logProb


	def appendLinearLayer(self, nNeurons, activationClass=nn.PReLU, dropoutRate=0.4, inputShape=None):
		if inputShape is None:
			if len(self.nn) == 0:
				inputShape = len(self.inputVariables)
			else:
				i=-1
				while abs(i) <= len(self.nn):
					try:
						if isinstance(self.nn[i], nn.BatchNorm1d):
							inputShape = self.nn[i].num_features
						else:
							inputShape = self.nn[i].out_features
						break
					except AttributeError:
						i -= 1
			if inputShape is None:
				raise Exception("inputShape missing")

		self.nn.append(nn.Linear(inputShape, nNeurons))
		if activationClass is not None:
			self.nn.append(activationClass())
		if dropoutRate is not None:
			self.nn.append(nn.Dropout(dropoutRate))


try:
	from modernplotting import mpplot
	from .._plottingUtils import plotSequentialModel
	def showModel(model: NeuralNetwork, style: mpplot.PlotterStyle) -> torchinfo.ModelStatistics:
		"""Show model information and create the model plot

		Args:
			model (NeuralNetwork): Model to be shown
			style (mpplot.PlotterStyle): Style for the model plot

		Returns:
			torchinfo.ModelStatistics: Summary of the model
		"""
		plotSequentialModel(model.nn, style, model.projectTag.replace('_', r'\_'))
		return torchinfo.summary(model, (1, len(model.inputVariables)), col_names=('input_size', 'output_size', 'num_params'))
except ImportError:
	def showModel(model: NeuralNetwork, style) -> torchinfo.ModelStatistics:
		raise NotImplementedError()
