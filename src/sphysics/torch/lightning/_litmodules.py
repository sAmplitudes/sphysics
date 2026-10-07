# coding: utf-8
"""
:Author: Stefan Wallner
:Description: Module for training with lightning
"""
# pylint: disable=arguments-differ,too-many-ancestors,assignment-from-none

from __future__ import absolute_import, print_function, division, annotations

from typing import Tuple, Union, Type, List
from abc import ABC, abstractmethod
import pathlib
import datetime
import itertools

import numpy as np
import torch
import torchmetrics
import lightning as L

from . import lr_schedulder

from ...utils import Logger

log = Logger("lightning")


class LitModuleBase(L.LightningModule, ABC):
	"""Base module for Lightning modules"""

	def __init__(self, learning_rate: float, lr_scheduler_para: dict, *args, save_hyperparameters: bool = True, **kwargs):
		"""
		All other arguments are used to construct the pytorch model.
		The given parameters will be stored as hyperparameters
		Args:
				learning_rate (float): Initial learning rate
				lr_scheduler_para (dict): Parameters for learning-rate scheduler
				save_hyperparameters (bool, optional): Save hyperparameters to training dir. Defaults to True
		"""
		super().__init__()
		if save_hyperparameters:
			self.save_hyperparameters()
		self.initial_learning_rate = learning_rate
		self.lr_scheduler_para = lr_scheduler_para

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.model(x)

	def training_step(self, batch: Tuple, batch_idx: int):
		self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_epoch=True, prog_bar=True)
		return self._shared_eval(batch, batch_idx, "train", metrics=False)

	def validation_step(self, batch: Tuple, batch_idx: int):
		self._shared_eval(batch, batch_idx, "val", metrics=True)

	def test_step(self, batch: Tuple, batch_idx: int):
		self._shared_eval(batch, batch_idx, "test", metrics=True)

	@abstractmethod
	def _shared_eval(
		self, batch: Tuple, batch_idx: int, prefix: str, metrics: bool
	) -> torch.Tensor:
		"""Common evaluation function that will be cassed by the training/validation/testing steps

		Args:
				batch (Tuple): Data batch, tuple as constructed in the data loader
				batch_idx (int): Index of the batch
				prefix (str): Prefix ('train', 'val', 'test')
				metrics (bool): Evaluate additional metrics

		Returns:
				torch.Tensor: Loss value
		"""
		raise NotImplementedError("Subclasses must implement this method")

	def configure_optimizers(self):
		"""Construct the optimizer and the learning-rate scheduler"""
		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.initial_learning_rate)
		lr_scheduler = self.configure_lr_scheduler(optimizer)
		if lr_scheduler is not None:
			return [optimizer], [
				{"scheduler": lr_scheduler, "interval": "epoch", "monitor": "val_loss"}
			]
		return optimizer

	def configure_lr_scheduler(
		self, _optimizer: torch.optim.Optimizer
	) -> torch.optim.lr_scheduler.LRScheduler:
		"""Construct the learning-rate schedulear. If returns None, no scheduler is used

		Args:
				optimizer (torch.optim.Optimizer): Optimizer

		Returns:
				torch.optim.lr_scheduler.LRScheduler: The used learning-rate scheduler
		"""
		return None


class LitModuleBinaryClassifier(lr_schedulder.ReduceOnPlateauMixin, LitModuleBase):
	"""Lightning module for binary classification problems.

	Expects a tuple of (x,y) values, where x is the input data and y is the class label, i.e. [0,1].
	"""

	def __init__(
		self,
		Model: Type[torch.Model],
		*args,
		learning_rate: float = 1e-2,
		lr_scheduler_para: dict = None,
		**kwargs,
	):
		"""_summary_

		All other arguments are used to construct the pytorch model.
		The given parameters will be stored as hyperparameters.
		Args:
				Model (Type[torch.Model]): Pytorch model class
				learning_rate (float): Initial learning rate
				lr_scheduler_para (dict): Parameters for learning-rate scheduler
		"""
		super().__init__(learning_rate, lr_scheduler_para, *args, **kwargs)
		self.model = Model(*args, **kwargs)
		self.loss = torch.nn.BCELoss()
		self.metrics = {"AUROC": torchmetrics.AUROC(task="binary")}

	def _shared_eval(self, batch: Tuple, batch_idx: int, prefix: str, metrics: bool):
		"""Common evaluation function that will be cassed by the training/validation/testing steps

		Args:
				batch (Tuple): Data batch, tuple as constructed in the data loader
				batch_idx (int): Index of the batch
				prefix (str): Prefix ('train', 'val', 'test')
				metrics (bool): Evaluate additional metrics

		Returns:
				torch.Tensor: Loss value
		"""
		x, y = batch
		y_hat = self.model(x)
		loss = self.loss(y_hat, y)
		self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True)
		if metrics:
			for metric_name, metric in self.metrics.items():
				metric_value = metric(y_hat, y)
				self.log(
					f"{prefix}_{metric_name}",
					metric_value,
					on_epoch=True,
					prog_bar=True,
				)
		return loss


class Training:
	"""Training object that handles the whole training"""

	def __init__(
		self,
		model: L.LightningModule,
		train_loader: torch.utils.data.DataLoader,
		val_loader: torch.utils.data.DataLoader,
		name: str,
		out_dir: Union[pathlib.Path, str],
		name_prefix_date: bool = False,
		checkpoint_every_n_epochs: int = 1,
		keep_to_k_epochs: int = -1,
	):
		"""
		Args:
				model (L.LightningModule): Lightning module
				train_loader (torch.utils.data.DataLoader): training data loader
				val_loader (torch.utils.data.DataLoader): validation data loader
				name (str): Name of the training.
				out_dir (Union[pathlib.Path, str]): Output dir where the training results are stored
				name_prefix_date (bool, optional):  If true, the name will be prefixed by `YYYY-mm-dd_HH-MM-SS`
				checkpoint_every_n_epochs (int, optional): Save checkpoint every n epochs
				keep_to_k_epochs (int, optional): Checkpoints of how many epochs to keep. Defaults to -1, meaning keep all epochs.
		"""
		self.model = model
		self.train_loader = train_loader
		self.val_loader = val_loader
		self.out_dir = pathlib.Path(out_dir)
		self.name = name
		if name_prefix_date:
			self.name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + self.name
		self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
		self.keep_top_k_epochs = keep_to_k_epochs
		log.info("Initialize training:")
		with log.indented():
			log.info(f"Name:		{self.name}")
			log.info(f"Output path: {self.out_dir}")

		self.init_trainer()

	def init_trainer(self):
		"""Initialize the trainer"""
		checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
			filename=self.name + "_{epoch:05d}",
			every_n_epochs=self.checkpoint_every_n_epochs,
			save_top_k=self.keep_top_k_epochs,  # keep all
		)

		self.trainer = L.Trainer(
			max_epochs=0,
			enable_progress_bar=True,
			log_every_n_steps=1000,
			logger=L.pytorch.loggers.CSVLogger(
				save_dir=self.out_dir / "training/",
				name=self.name,
				flush_logs_every_n_steps=1000,
			),
			callbacks=[checkpoint_callback],
		)

	def fit(
		self,
		max_epochs: int,
		resume_from_checkpoint: Union[bool, str, pathlib.Path] = False,
	):
		"""Perform training

		Args:
				max_epochs (int): Maximum number of epochs to train in total, i.e. including previous calls to `fit`.
				resume_from_checkpoint (bool, optional): Load from previous checkpoint. Defaults to True.
														 If True, use latest checkpoint from latest version.
														 If string or path use the given checkpoint.
														 If not restored from checkpoint, the learning rate will start again from the initial value.

		Returns:
				_type_: _description_
		"""
		self.trainer.fit_loop.max_epochs = max_epochs

		resume_ckpt_path = None
		if resume_from_checkpoint is not False:
			if isinstance(resume_from_checkpoint, (str, pathlib.Path)):
				resume_ckpt_path = pathlib.Path(resume_from_checkpoint)
			elif resume_from_checkpoint is True:
				# find latest version with checkpoints
				latest_versions = self.getVersionDirs()
				if latest_versions:
					resume_ckpt_paths = list(
						sorted((latest_versions[-1] / "checkpoints").glob("*.ckpt"))
					)
					if resume_ckpt_paths:
						resume_ckpt_path = resume_ckpt_paths[-1]
			else:
				log.raiseException(
					Exception, "Unknown input type of `resume_from_checkpoint`"
				)

		return self.trainer.fit(
			model=self.model,
			train_dataloaders=self.train_loader,
			val_dataloaders=self.val_loader,
			ckpt_path=resume_ckpt_path,
		)

	def getVersions(self) -> List[int]:
		"""Get list of versions of the current training name ordered from newest to latest

		Returns:
				List[int]: list of versions of the current training name ordered from newest to latest
		"""
		curr_version_dir = pathlib.Path(self.trainer.logger.log_dir)
		# find latest version with checkpoints
		return np.unique(
			list(
				sorted(
					[
						int(vdir.parent.parent.stem.replace("version_", ""))
						for vdir in curr_version_dir.parent.glob(
							"version_*/checkpoints/*.ckpt"
						)
					]
				)
			)
		)

	def getVersionDirs(self) -> List[pathlib.Path]:
		"""Get list of paths to version folders of the current training name ordered from newest to latest

		Returns:
				List[int]: List of paths to version folders of the current training name ordered from newest to latest
		"""
		curr_version_dir = pathlib.Path(self.trainer.logger.log_dir)
		return [curr_version_dir.parent / f"version_{v}" for v in self.getVersions()]


	def getCheckpoints(self) -> List[pathlib.Path]:
		"""Get checkpoints

		Returns:
			List[pathlib.Path]: List of checkpoints ordered by version and epoch.
		"""
		return list(itertools.chain(
			*[
				list(sorted((version_dir / "checkpoints").glob("*.ckpt")))
				for version_dir in self.getVersionDirs()
			]
		))
