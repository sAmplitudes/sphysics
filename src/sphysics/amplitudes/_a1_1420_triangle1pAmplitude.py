# coding: utf-8
'''
:Author: Godo Kurten
:Description: Triangle amplitude for a1(1420), Created on Friday 12.12.25
'''

from __future__ import absolute_import, print_function, division, annotations

import os

import numpy as np
import tensorflow as tf

from ._utils import TabulatedAmplitude
from ._resonance import breitWigner
from .._constants import Constants as C

# Get triangle Look-up table
triangleTableValues = TabulatedAmplitude(os.path.join(os.environ['SPHYSICS'], 'data', 'amplitudes', 'K0K0barspinnolam.csv'))

def a1_1420_triangleAmplitude(mass: float|np.ndarray|tf.Tensor) -> float|np.ndarray|tf.Tensor:
	"""Triangle amplitude of the a1_1420 resonance.
	This amplitude was taken from COMPASS, PRL 127, 082501 (2021)
	https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.082501

	Args:
		mass (float | np.ndarray | tf.Tensor): Mass where the amplitude should be evaluated

	Returns:
		float|np.ndarray|tf.Tensor: Triangle amplitude of the a1_1420 resonance at the given mass points
	"""
	# Make mass an array if it is a float or integer
	if isinstance(mass, (int, float)):
		mass = np.array(mass)

	# Evaluate triangle amplitude at given mass points
	triangleAmp = triangleTableValues(mass)

	return triangleAmp

def a1_1420_triangleMulA1Amplitude(mass: float|np.ndarray|tf.Tensor, massA11260: float = C.M['a1(1260)-'], widthA11260: float = C.G['a1(1260)-']) -> float|np.ndarray|tf.Tensor:
	"""This method calculates the triangle amplitude of the a1(1420) multiplied by the breitWigner amplitude for the a1(1260).
	The mass and width for the a1(1260) breitWigner are taken from the Constants class.
	We do not normalize this amplitude!!!
	The triangle amplitude of the a1(1420) is taken from COMPASS, PRL 127, 082501 (2021)
	https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.127.082501

	Args:
		mass (float | np.ndarray | tf.Tensor): Mass where the amplitude should be evaluated
		massA11260 (float, optional): Mass of the a1(1260) resonance. Defaults to C.M['a1(1260)-'].
		widthA11260 (float, optional): Width of the a1(1260) resonance. Defaults to C.G['a1(1260)-'].

	Returns:
		float|np.ndarray|tf.Tensor: Triangle amplitude of the a1(1420) multiplied by the breitWigner amplitude for the a1(1260)
	"""
	# Make mass an array if it is a float or integer
	if isinstance(mass, (int, float)):
		mass = np.array(mass)

	# Get Breit-Wigner Amplitude of a1(1260)
	if isinstance(mass, tf.Tensor):
		mA1 = tf.constant(massA11260, dtype=tf.float64)
		gA1 = tf.constant(widthA11260, dtype=tf.float64)
	else:
		mA1 = massA11260
		gA1 = widthA11260
	breitWignerAmp = breitWigner(mass, C.M.pi, C.M['rho(770)0'], 0, mA1, gA1)

	# Get triangle Look-up table
	triangleAmp = a1_1420_triangleAmplitude(mass)

	# Calculate full amplitude
	amp = breitWignerAmp * triangleAmp

	return amp
