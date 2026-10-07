# coding: utf-8
"""
:Author: Stefan Wallner
:Description: Mixins for learning rate scheduler
"""

from __future__ import absolute_import, print_function, division, annotations

import torch


class ReduceOnPlateauMixin:
	"""Mixin to use ReduceLROnPlateau learning rate scheduler"""

	def configure_lr_scheduler(
		self, optimizer: torch.optim.Optimizer
	) -> torch.optim.lr_scheduler.LRScheduler:
		'''Configure ReduceLROnPlateau scheduler with default parameters. Defaults can be overridden by custom parameters.
		'''
		kwargs = {
			"mode": "min",
			"threshold": 1e-4,
			"threshold_mode": "rel",
			"factor": 0.2,
			"patience": 10,
			"min_lr": 1e-5,
		}
		if self.lr_scheduler_para:
			kwargs.update(self.lr_scheduler_para)
		return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)


class ExponentialLRMixin:
	"""Mixin to use ExponentialLR learning rate scheduler"""

	def configure_lr_scheduler(
		self, optimizer: torch.optim.Optimizer
	) -> torch.optim.lr_scheduler.LRScheduler:
		'''Configure ExponentialLR scheduler.
		'''
		return torch.optim.lr_scheduler.ExponentialLR(
			optimizer, **self.lr_scheduler_para
		)
