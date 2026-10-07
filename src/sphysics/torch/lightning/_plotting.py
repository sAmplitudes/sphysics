# coding: utf-8
"""
:Author: Stefan Wallner
:Description: Module for plotting training results
"""
from __future__ import absolute_import, print_function, division, annotations

import pathlib
import pandas as pd

import modernplotting
import modernplotting.mpplot

from ._litmodules import Training


def plotMetric(
	training: Training,
	metric: str,
	style: modernplotting.mpplot.PlotterStyle,
	merge_versions: bool = False,
	min_max="min",
	zoom_in: float = None
) -> modernplotting.mpplot.MPPlot1D:
	"""Plot a specific metric from the training results.

	Args:
		training (Training): The Training object containing the training results.
		metric (str): The name of the metric to plot.
		style (modernplotting.mpplot.PlotterStyle): The plotting style to use.
		merge_versions (bool, optional): Whether to merge data from all versions. Defaults to False.
		min_max (str, optional): Plot either the minimum or maximum value. Must be "min" or "max". Defaults to "min".
		zoom_in (float, optional): Factor to zoom in on the plot. If None, no zoom is applied. Defaults to None.

	Returns:
		modernplotting.mpplot.MPPlot1D: The plot.
	"""

	# load data
	if merge_versions:
		dfs = []
		for version_dir in training.getVersionDirs():
			metric_file = version_dir / "metrics.csv"
			if metric_file.exists():
				dfs.append(pd.read_csv(metric_file))
		df = pd.concat(dfs)
	else:
		df = pd.read_csv(pathlib.Path(training.trainer.logger.log_dir) / "metrics.csv")
	if metric == "loss":
		metric_train = "train_loss_epoch"
	elif metric == "lr":
		metric_train = "lr_epoch"
	else:
		metric_train = f"train_{metric}"
	metric_val = f"val_{metric}"
	has_train = metric_train in df
	has_val = metric_val in df
	if not has_train and not has_val:
		return None
	if has_train and has_val:
		metric = metric_val
		plot = style.getPlot1DTwin()
		df.plot(
			x="epoch",
			y=metric_train,
			kind="scatter",
			ax=plot.axesR,
			label="training",
			legend=False,
		)
		df.plot(
			x="epoch",
			y=metric_val,
			kind="scatter",
			ax=plot.axes,
			color=style.colorScheme.orange,
			label="validation",
			legend=False,
			marker="x",
		)
		plot.setYAxesColorR(style.getP1DColorCyclerColor(0))
		color = style.colorScheme.orange
	else:

		metric = metric_train if has_train else metric_val
		kwargs = {}
		color = style.getP1DColorCyclerColor(0)
		plot = style.getPlot1D()
		if has_train:
			kwargs["color"] = style.colorScheme.orange
			color = kwargs["color"]
		df.plot(
			x="epoch", y=metric, kind="scatter", ax=plot.axes, legend=False, **kwargs
		)

	plot.setYAxesColor(color)
	if min_max:
		plot.plotHorizontalLine(
			df[metric].min() if min_max == "min" else df[metric].max(),
			color=color,
			zorder=1.3,
		)
	if zoom_in is not None and min_max:
		if min_max == "min":
			if has_train and has_val:
				plot.setYlimR(ymin=df[metric_train].min()-zoom_in*0.05, ymax=df[metric_train].min() + zoom_in)
				plot.setYlim(ymin=df[metric_val].min()-zoom_in*0.05, ymax=df[metric_val].min() + zoom_in)
			else:
				plot.setYlim(ymin=df[metric].min() - zoom_in*0.05, ymax=df[metric].min() + zoom_in)
		else:
			if has_train and has_val:
				plot.setYlimR(ymin=df[metric_train].max() - zoom_in, ymax=df[metric_train].max() + zoom_in*0.05)
				plot.setYlim(ymin=df[metric_val].max() - zoom_in, ymax=df[metric_val].max() + zoom_in*0.05)
			else:
				plot.setYlim(ymin=df[metric].max() - zoom_in, ymax=df[metric].max() + zoom_in*0.05)

	return plot
