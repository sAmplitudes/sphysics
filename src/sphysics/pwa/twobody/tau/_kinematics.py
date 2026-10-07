# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Skeleton kinematics for tau -> two-body decays
'''

from __future__ import absolute_import, print_function, division

import numpy as np

from .._kinematics import Kinematics


class Tau2Kinematics(Kinematics):
	'''
	Tau-specific two-body kinematics.
	'''

	def __init__(self, p1: np.ndarray = None, p2: np.ndarray = None, momentaInCMS: bool = False) -> None:
		super().__init__(p1=p1, p2=p2, momentaInCMS=momentaInCMS)
