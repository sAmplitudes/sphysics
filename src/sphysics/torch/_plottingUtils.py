# coding: utf-8
'''
Created on Wednesday 29 06 2022
Author: Stefan Wallner
Description: Utility functions for torch
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
from torch import nn
from modernplotting import mpplot




def plotSequentialModel(model: nn.Sequential, style: mpplot.PlotterStyle, title: str=None) -> mpplot.MPPlot1D:
	"""Create a plot that visualizes the layers of a sequential model of linear layers

	Args:
		model (nn.Sequential): Model to be plotted
		style (mpplot.PlotterStyle): Style for the plot
		title (str, optional): Title for the plot. Defaults to None.

	Returns:
		mpplot.MPPlot1D: The finished plot
	"""
	widths = [model[0].in_features]
	for i, layer in enumerate(model):
		if isinstance(layer, nn.Linear):
			widths.append(layer.out_features)

	plot = style.getPlot1D(figHeight=1.3, figWidth=0.37*len(widths))

	for i, width in enumerate(widths):
		if i == 0:
			color = style.colorScheme.red
		elif i == len(widths)-1:
			color = style.colorScheme.green
		else:
			color = style.colorScheme.blue
		plot.plot((i,i), (-width/2.,width/2.), color=color, lw=5)
		plot.text(i-0.35, 6/512*np.max(widths), f"{width}", rotation=90, va='center', ha='center', fontsize=style.legendFontSize*0.7, color=color)

	plot.setXticks(np.arange(len(widths)))
	plot.setXtickLabels(["input"] + [ str(i) for i in range(1, len(widths)-1) ] + ["output"])
	plot.setXshowMinorTicks(False)
	plot.axes.spines.right.set_visible(False)
	plot.axes.spines.top.set_visible(False)
	plot.axes.spines.left.set_visible(False)
	plot.axes.get_yaxis().set_visible(False)
	if title is not None:
		plot.addTitleLeft(title)
	plot.finish()
	return plot
