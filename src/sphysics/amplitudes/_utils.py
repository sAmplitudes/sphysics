# coding: utf-8
'''
:Author: Stefan Wallner

:Description: Utility functions and classes for amplitudes, Created on Thursday 05 05 2022
'''

from __future__ import absolute_import, print_function, division, annotations

from typing import Callable, Any
import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf

from .. import math


class LambdaF:
	"""Lambda class that wraps a function that depends on variables and parameters

	This class wraps a function `fcn`, whose FIRST arguments are the actual variable and all OTHER arguments are constant parameters,
	to a callable, whose arguments are only the variables.
	The constant parameters are set in the `LambdaM` initialization (see constructor).

	:param fcn: The actual function that should be called
	"""
	def __init__(self, fcn: Callable[..., Any], *args, **kwargs):
		"""
		:param fcn: The actual function that should be called
		*args and **kwargs are forwarded to the call of `fcn` as arguments in addition to the arguments of the call operator
		"""
		self._fcn = fcn
		self._args = args
		self._kwargs = kwargs


	def __call__(self, *args, **kwargs):
		kwargs.update(self._kwargs)
		args += self._args
		return self._fcn(*args, **kwargs)



class TabulatedAmplitude:


	def __init__(self, filePath: str, delimiter: str = ',', comments: str = '#') -> None:
		self.masses = None
		self.amplitudes = None
		self.filePath = filePath
		self.delimiter = delimiter
		self.comments = comments
		self._loadData()
		differences = self.masses[1:] - self.masses[:-1]
		is_equally_spaced = np.all(np.abs(differences - differences[0]) < 1e-9)
		if not is_equally_spaced:
			raise Exception("The masses in the tabulated amplitude file must be equally spaced for interpolation.")


	def _loadData(self) -> None:
		data = np.loadtxt(self.filePath, delimiter=self.delimiter, comments=self.comments)
		self.masses = data[:,0]
		self.amplitudes = data[:,1] + 1j*data[:,2]


	def __call__(self, mass: float | np.ndarray | tf.Tensor) -> Any:
		if isinstance(mass, tf.Tensor):
			real = tfp.math.interp_regular_1d_grid(x=mass,x_ref_min=min(self.masses),x_ref_max=max(self.masses),y_ref=math.real(self.amplitudes))
			imag = tfp.math.interp_regular_1d_grid(x=mass,x_ref_min=min(self.masses),x_ref_max=max(self.masses),y_ref=math.imag(self.amplitudes))
			interp = tf.complex(real, imag)
		else:
			interp = np.interp(mass, self.masses, self.amplitudes)

		return interp

def prepare_mass_tensor(mass):
	"""
	When mass or s is a single number, i.e. int or float, it convert it to a numpy array with a single element.
	the array/tensor is then cast to complex.


	Args:
		mass (int | float | np.ndarray | tf.Tensor): Is the (squared)-mass of a two-body system.

	Returns:
		np.ndarray | tf.Tensor: Same (squared)-mass cast to complex array/tensor.
	"""
	if isinstance(mass, (float, int)):
		mass = np.array([mass])
	mass = math.castToComplex(mass)
	return mass
