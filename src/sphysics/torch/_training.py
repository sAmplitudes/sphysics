# coding: utf-8
'''
Created on Tuesday 31 05 2022

Helper functions and classes for training pytorch neural networks
'''

from __future__ import absolute_import, print_function, division, annotations

import os
import glob
import collections
import shutil
import numpy as np

import torch
from torch.utils.data import DataLoader

from ..utils import Logger
from ..utils import Timer
from . import metrics

log = Logger("torch")


class TrainingHistory:
	"""Class to store the training history (loss, metrics, ....)
	"""
	def __init__(self):
		self.epochs = []
		self.metricsTraining = collections.OrderedDict()
		self.metricsValidation = collections.OrderedDict()

	@property
	def maxEpoch(self):
		'''Return the last epoch, defaults to 0.
		'''
		return max(self.epochs) if self.epochs else 0

	def appendMetrics(self, epoch, metricsTraining: collections.OrderedDict, metricsValidation: collections.OrderedDict) -> None:
		"""Append metrics for one epoch

		Args:
			metricsTraining (collections.OrderedDict): Metrics for the training sample
			metricsValidation (collections.OrderedDict): Metrics for the validation sample
		"""
		self.epochs.append(epoch)
		if not self.metricsTraining:
			for metric in metricsTraining:
				self.metricsTraining[metric] = [metricsTraining[metric]]
			for metric in metricsValidation:
				self.metricsValidation[metric] = [metricsValidation[metric]]
		else:
			for metric in metricsTraining:
				self.metricsTraining[metric].append(metricsTraining[metric])
			for metric in metricsValidation:
				self.metricsValidation[metric].append(metricsValidation[metric])


	def __iadd__(self, other: TrainingHistory) -> TrainingHistory:
		"""Add the `other` TrainingHistory to this history.

		Args:
			other (TrainingHistory): Will be added to this training history

		Returns:
			TrainingHistory: This training history after adding
		"""
		self.epochs += other.epochs
		for metrics, metricsOther in zip(( self.metricsTraining,  self.metricsValidation),
										 (other.metricsTraining, other.metricsValidation)):
			metricNames = list(metrics.keys())
			metricNames += [metric for metric in metricsOther.keys() if metric not in metricNames]
			for metric in metricNames:
				if metric not in metrics:
					metrics[metric] = metricsOther[metric].copy()
				elif metric in metricsOther:
					metrics[metric] += metricsOther[metric]
		return self



class Trainer:
	'''Handle the training, validation, metric tracking, and saving of a pytorch model.
	'''
	def __init__(self, model, lossFunction, optimizer, lrScheduler=None, verbose=None):
		self.model = model
		self._lossFunction = lossFunction
		self._optimizer = optimizer
		self._lrScheduler = lrScheduler
		self._verbose = verbose

		self.timer = None
		self.projectDir = None
		self.datasetTraining= None
		self.datasetValidation= None
		self.datasetTrainingLoader = None
		self.datasetValidationLoader = None
		self.metricClasses = None


	def trainEpoch(self, dataloader):
		'''Training the model for one epoch on 'dataloader'.

		:param dataloader: Training data
		:return: Training Loss
		:rtype: float
		'''
		nBatches = len(dataloader)
		self.model.train()
		if self._verbose:
			log.initProgress(nBatches)
		totalLoss = metrics.Loss()
		for batch, (x, y, _) in enumerate(dataloader):

			# Compute prediction error
			pred = self.model(x)
			loss = self._lossFunction(pred, y)

			# Backpropagation
			# self._optimizer.zero_grad()
			for param in self.model.parameters():
				param.grad = None
			loss.backward()
			self._optimizer.step()

			totalLoss(loss.cpu().detach().numpy())

			if self._verbose and batch % int(nBatches/5) == 0:
				log.printProgress(batch+1, nBatches)
		if self._lrScheduler is not None:
			self._lrScheduler.step()
		if self._verbose:
			log.printProgress(nBatches, nBatches)
			log.finishProgress()
		return totalLoss.compute()



	def testEpoch(self, dataloader):
		'''Evaluation on 'dataloader'.

		:param dataloader: Validation data
		:return: Validation loss
		'''
		metricInstances = [metricClass() for metricClass in self.metricClasses]

		self.model.eval()
		with torch.no_grad():
			for x, y, _ in dataloader:
				predictedLogProb = self.model(x)
				predictedProb = torch.exp(predictedLogProb)
				predictedBest = predictedLogProb.argmax(1)
				loss = self._lossFunction(predictedLogProb, y).item()
				kwargs = {'loss': loss, 'labels': y, 'predictedLogProb': predictedLogProb, 'predictedBest': predictedBest, 'predictedProb': predictedProb}
				for metric in metricInstances:
					metric(**kwargs)
		return collections.OrderedDict([(metric.name, metric.compute()) for metric in metricInstances])



	def checkpoint(self, epoch):
		'''Save model, optimizer and learning rate states.

		:param epoch: Epoch number, checkpoint is labeled by `epoch`
		:type epoch: int
		'''
		checkpointFile = os.path.join(self.projectDir, f'epoch-{epoch:06d}')
		data = {
			'model': self.model,
			'optimizer_state_dict': self._optimizer.state_dict(),
			'lrScheduler_state_dict': self._lrScheduler.state_dict() if self._lrScheduler else None,
		}
		torch.save(data, checkpointFile)


	def saveData(self):
		'''Save the training and validation datasets in 'data.pkl'
		'''
		if not os.path.exists(self.projectDir):
			os.makedirs(self.projectDir)
		torch.save({'datasetTraining': self.datasetTraining, 'datasetValidation': self.datasetValidation},
		           os.path.join(self.projectDir, 'data.pkl'))


	def trainLoopBody(self, epoch: int) -> None:
		'''Perform  training and validation step for the specified `epoch`.

		:param epoch: Epoch number
		:type epoch: int
		'''
		if self._verbose:
			log.info(f"Epoch {epoch}")
		with log.indented(), self.timer.interval("Epoch"):
			trainLoss = self.trainEpoch(self.datasetTrainingLoader)
			metricsTrainingEpoch = collections.OrderedDict([('loss', trainLoss)])
			metricsValidationEpoch = self.testEpoch(self.datasetValidationLoader)
			out=[]
			for _, metricInstances in zip(("train", "val"), (metricsTrainingEpoch, metricsValidationEpoch)):
				for metricName in metricInstances:
					out.append(f"{metricInstances[metricName]:<10.3g}")
					# writer.add_scalar(f'{metricName}/{tag}', metrics[metricName], epoch)
			if self._verbose:
				log.info("".join(out))
			self.model.history.appendMetrics(epoch, metricsTrainingEpoch, metricsValidationEpoch)
			self.checkpoint(epoch)


	def train(self, datasetTraining, datasetValidation, epochs, batchSize, projectDir,
	          approxTrainMetrics=True, metricClasses=None) -> TrainingHistory:
		'''Run full training loop over number of 'epochs', including training, validation, logging and creating checkpoints.

		:param datasetTraining: Training dataset
		:param datasetValidation: Validation dataset
		:param epochs: Number of epoch
		:type epochs: int
		:param batchSize: Batch size for training
		:type batchSize: int
		:param projectDir: Directory for saving the checkpoints
		:type projectDir: str
		:param approxTrainMetrics: If True, only training loss is computed during training, otherwise full metrixs are computed, defaults to True
		:type approxTrainMetrics: bool, optional
		:param metricClasses: List of metric classes, defaults to None
		:type metricClasses: list, optional
		:return: `TrainingHistory` containing training and validation metrics
		:rtype: TrainingHistory
		'''
		self.metricClasses = metricClasses
		self.projectDir = projectDir
		self.datasetTraining = datasetTraining
		self.datasetValidation = datasetValidation

		self.timer = Timer()
		self.timer.time("start")

		self.saveData()

		# writer = torch.utils.tensorboard.SummaryWriter()
		with log.indented():
			tagheader = []
			headers = []
			for tag in ['training', 'validation']:
				metricInstances = ( [metricClass() for metricClass in self.metricClasses] if not approxTrainMetrics or tag != 'training' else [metrics.Loss()] )
				tagheader.append('{0:{1}s}'.format(tag, 10*len(metricInstances)))
				for metric in metricInstances :
					headers.append(f'{metric.name:<10s}')
			if self._verbose:
				log.info("".join(tagheader))
				log.info("".join(headers))
		self.datasetTrainingLoader   = DataLoader(datasetTraining, batch_size=batchSize, shuffle=True)
		self.datasetValidationLoader = DataLoader(datasetValidation, batch_size=2048)
		for epoch in range(self.model.history.maxEpoch+1, self.model.history.maxEpoch+1+epochs):
			self.trainLoopBody(epoch)
		# writer.flush()
		self.timer.time("end")
		if self._verbose:
			log.success("Done!")
			self.timer.printStatistics()
		return self.model.history

	@classmethod
	def loadCheckpoint(cls, projectDir, device=None, checkpoint = None, bestOfLast=None, verbose=True):
		epochBest = None
		fullModel = None

		if checkpoint is None and bestOfLast is not None:
			fullModel = cls.loadCheckpoint(projectDir, device=device, verbose=verbose).model
			fullAUROC = np.array(fullModel.history.metricsValidation['AUROC'])
			nLast = bestOfLast

			nLast = min(nLast, fullAUROC.size)
			epochBest = fullModel.history.epochs[np.argmax(fullAUROC[-nLast:]) +
												(fullAUROC.size - nLast)]
			checkpoint = epochBest
			if verbose:
				log.success(
					f"Best epoch is {checkpoint} with AUROC {fullAUROC[np.argwhere(np.array(fullModel.history.epochs)==epochBest)[0]][0]}"
				)
		if checkpoint is None:
			checkpointFile = sorted(glob.glob(os.path.join(projectDir, 'epoch-*')))[-1]
		else:
			checkpointFile = os.path.join(projectDir, f'epoch-{checkpoint:06d}')

		if not os.path.isfile(checkpointFile):
			log.raiseException(Exception, "Cannot find '{0}'", checkpointFile)
		if verbose:
			log.info(f'Load checkpoint "{checkpointFile}"')
		data = torch.load(checkpointFile, map_location=device)

		model = data['model']
		projectTag = model.projectTag
		if fullModel is not None:
			model.history = fullModel.history

		optimizer = torch.optim.Adam(model.parameters())
		optimizer.load_state_dict(data['optimizer_state_dict'])

		lossFunction = torch.nn.NLLLoss()

		if data['lrScheduler_state_dict']:
			lrScheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
			lrScheduler.load_state_dict(data['lrScheduler_state_dict'])
		else:
			lrScheduler = None

		trainer = cls(model, lossFunction, optimizer, lrScheduler, verbose=verbose)
		if verbose:
			log.emph(f"Project tag: {projectTag}; epoch: {epochBest if epochBest is not None else model.history.maxEpoch}")

		if epochBest is not None:
			trainer.epochBestattribute=epochBest #pylint: disable=attribute-defined-outside-init
		return trainer


def saveBestEpoch(allModelDirectory,bestEpochDirectory,projectName):
	"""It saves the best and the last epoch of all in the chosen directory.
	The criterion to define the best epoch is the AUROC of the validation sample.

	:param allModelDirectory: Name of the directory were you have all your models (with all the epochs)
	:type allModelDirectory: str
	:param bestEpochDirectory: Name of the directory were you want to save the best epoch
	:type bestEpochDirectory: str
	:param projectName: Name of the project that you want to save the best epoch
	:type bestEpochDirectory: str

	:Example:
		allModelDirectory="/path/to/allModels"
		bestEpochDirectory="/path/to/bestEpochModels"
		projectName= "myProject"
	"""


	m= Trainer.loadCheckpoint(os.path.join(allModelDirectory, projectName), bestOfLast=10000000, device='cpu') # Always use CPU as it should always work
	bestEpoch=m.epochBestattribute

	if os.path.exists(os.path.join(bestEpochDirectory, projectName)):
		shutil.rmtree( os.path.join(bestEpochDirectory, projectName))
	os.mkdir (os.path.join(bestEpochDirectory, projectName))

	# save last epoch
	epochString=f'epoch-{m.model.history.maxEpoch:06d}'
	fromEpochFile=os.path.join(allModelDirectory, projectName,epochString)
	toEpochFile=os.path.join(bestEpochDirectory, projectName,epochString)
	shutil.copy2(fromEpochFile, toEpochFile)

	# save best epoch
	epochString=f'epoch-{bestEpoch:06d}'
	fromEpochFile=os.path.join(allModelDirectory, projectName,epochString)
	toEpochFile=os.path.join(bestEpochDirectory, projectName,epochString)
	shutil.copy2(fromEpochFile, toEpochFile)
	return toEpochFile
