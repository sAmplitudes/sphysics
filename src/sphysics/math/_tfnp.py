# coding: utf-8
'''
Functions that can take tensorflow as well as numpy arguments

:Author: Stefan Wallner
'''

# pylint: disable=redefined-builtin,invalid-name

from __future__ import annotations

import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly
import tensorflow as tf

from .. import utils

logger = utils.Logger('math')

#pylint: disable=no-else-return

def einsum(equation, *inputs, **kwargs):
	'''Computes Einstein summation for tensors.

	:param equation: Specifies the subscripts for the summation.
	:type equation: str
	:return: The result of the einsum operation, computed using either NumPy or TensorFlow.
	:rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(inputs[0], tf.Tensor):
		return tf.einsum(equation, *inputs, **kwargs)
	else:
		return np.einsum(equation, *inputs, **kwargs)


def expand_dims(input: tf.Tensor|np.ndarray, axis: int, **kwargs):   # pylint: disable=invalid-name
	'''Expands the shape of an input tensor by adding a new axis at the specified position.

    :param input: Input tensor to expand
    :type input: tf.Tensor | np.ndarray
    :param axis: Position where the new axis is placed
    :type axis: int
    :return: A tensor with an additional dimension along the specified axis
    :rtype: tf.Tensor | np.ndarray
    '''
	if isinstance(input, tf.Tensor):
		return tf.expand_dims(input, axis, **kwargs)
	else:
		return np.expand_dims(input, axis, **kwargs)


def abs(tensor, **kwargs):
	'''Computes the absolute value of a tensor.

	:param tensor: The input tensor.
	:type tensor: np.ndarray or tf.Tensor
	:return: The absolute value of the tensor.
	:rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.math.abs(tensor, **kwargs)
	else:
		return np.abs(tensor, **kwargs)


def max(tensor, **kwargs):
	'''Computes the maximum value of a tensor.

	:param tensor: The input array or tensor
	:type tensor: np.ndarray or tf.Tensor
	:return: Maximum value
	:rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.math.reduce_max(tensor, **kwargs)
	else:
		return np.amax(tensor, **kwargs)

def min(tensor, **kwargs):
	'''Computes the minimum value of a tensor.

	:param tensor: The input tensor
	:type tensor: np.ndarray or tf.Tensor
	:return: Minimum value
	:rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.math.reduce_min(tensor, **kwargs)
	else:
		return np.amin(tensor, **kwargs)


def mean(tensor, **kwargs):
	'''Computes the mean value of a tensor.

    :param tensor: The input tensor
    :type tensor: np.ndarray or tf.Tensor
    :return: Mean value
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.reduce_mean(tensor, **kwargs)
	else:
		return np.mean(tensor, **kwargs)


def sin(tensor, **kwargs):
	'''Computes the sine of a tensor, element-wise.

    :param tensor: The input tensor
    :type tensor: np.ndarray or tf.Tensor
    :return: Sine values
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.sin(tensor, **kwargs)
	else:
		return np.sin(tensor, **kwargs)


def cos(tensor, **kwargs):
	'''Computes the cosine of a tensor, element-wise.

    :param tensor: The input tensor
    :type tensor: np.ndarray or tf.Tensor
    :return: Cosine values
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.cos(tensor, **kwargs)
	else:
		return np.cos(tensor, **kwargs)

def arctan(tensorA, **kwargs):
	'''Computes the trigonometric inverse tangent of a tensor, element-wise.

    :param tensorA: The input tensor
    :type tensorA: np.ndarray or tf.Tensor
    :return: Arctan values
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensorA, tf.Tensor):
		return tf.math.atan(tensorA, **kwargs)
	else:
		return np.arctan(tensorA, **kwargs)

def arctan2(tensorA, tensorB, **kwargs):
	'''Computes the trigonometric inverse tangent of two tensors, element-wise.

    :param tensorA: First input tensor.
    :type tensorA: np.ndarray or tf.Tensor
    :param tensorB: Second input tensor.
    :type tensorB: np.ndarray or tf.Tensor
    :return: Arctan2 values.
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensorA, tf.Tensor):
		return tf.math.atan2(tensorA, tensorB, **kwargs)
	else:
		return np.arctan2(tensorA, tensorB, **kwargs)

def exp(tensor, **kwargs):
	'''Computes the exponential of a tensor, element-wise.

    :param tensor: The input tensor
    :type tensor: np.ndarray or tf.Tensor
    :return: Exponential values
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.exp(tensor, **kwargs)
	else:
		return np.exp(tensor, **kwargs)

def log(tensor, **kwargs):
	'''Computes the logarithm of a tensor, element-wise.

    :param tensor: The input tensor
    :type tensor: np.ndarray or tf.Tensor
    :return: Logarithm values
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.log(tensor, **kwargs)
	else:
		return np.log(tensor, **kwargs)

def pow(tensor, power, **kwargs):
	'''Computes the power of a tensor, element-wise.

    :param tensor: The base tensor
    :type tensor: np.ndarray or tf.Tensor
    :param power: The exponent
    :type power: float or int
    :return: Tensor with value raised to a given power
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.pow(tensor, power, **kwargs)
	else:
		return np.power(tensor, power, **kwargs)


def sum(tensor, **kwargs):
	'''Computes the sum of all elements in a tensor.

    :param tensor: The input tensor
    :type tensor: np.ndarray or tf.Tensor
    :return: Sum of elements
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.reduce_sum(tensor, **kwargs)
	else:
		return np.sum(tensor, **kwargs)

def transpose(tensor, axes, **kwargs):
	'''Transposes the dimensions of a tensor.

    :param tensor: The input tensor.
    :type tensor: np.ndarray or tf.Tensor
    :param axes: Permutation of dimensions.
    :type axes: tuple or list
    :return: Transposed tensor.
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.transpose(tensor, axes, **kwargs)
	else:
		return np.transpose(tensor, axes, **kwargs)

def real(tensor):
	'''Returns the real part of a complex (or real) tensor.

    :param tensor: The input tensor.
    :type tensor: np.ndarray or tf.Tensor
    :return: The real part of the input tensor.
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.real(tensor)
	else:
		return np.real(tensor)

def imag(tensor):
	'''Returns the imaginary part of a complex (or real) tensor.

    :param tensor: The input tensor.
    :type tensor: np.ndarray or tf.Tensor
    :return: The imaginary part of the input tensor.
    :rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.math.imag(tensor)
	else:
		return np.imag(tensor)

def conjugate(tensor, **kwargs):
	'''Computes the complex conjugate of a tensor, element-wise.

    :param tensor: The input tensor.
    :type tensor: np.ndarray or tf.Tensor
    :return: Charge conjugate of a tensor.
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(tensor, tf.Tensor):
		return tf.math.conj(tensor, **kwargs)
	else:
		return np.conjugate(tensor, **kwargs)

def empty(shape, refTensor, **kwargs):
	'''Creates a new tensor of the given shape.

    - If `refTensor` is a TensorFlow tensor, the function returns a new tensor initialized with zeros.
    - If `refTensor` is a Numpy ndarray, the function returns an uninitialized NumPy tensor.

    :param shape: Shape of the new tensor
    :type shape: int or list of int
    :param refTensor: If a tf.Tensor is given a tensor is returned, otherwise a numpy.ndarray
    :type refTensor: tf.Tensor or np.ndarray
    :return: A new tensor, filled with `fill_value`
    :rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(refTensor, tf.Tensor):
		return full(shape, 0., refTensor, **kwargs)
	else:
		return np.empty(shape, **kwargs)

def empty_like(refTensor, **kwargs):
	shape = refTensor.shape
	if 'dtype' not in kwargs:
		kwargs['dtype'] = refTensor.dtype
	return empty(shape, refTensor, **kwargs)

def full(shape, fill_value, refTensor, **kwargs):
	'''Creates a new tensor of the given shape, filled with a specified value.

    :param shape: Shape of the new tensor
    :type shape: int or list of int
    :param fill_value: Value to fill the tensor with
    :param refTensor: Reference tensor that determines whether the output is a TensorFlow tensor or a NumPy ndarray.
    :type refTensor: tf.Tensor or np.ndarray
    :return: A new tensor, filled with `fill_value`
    :rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(refTensor, tf.Tensor):
		return tf.ones(shape, **kwargs) * fill_value
	else:
		return np.full(shape, fill_value, **kwargs)

def full_like(refTensor, fill_value, **kwargs):
	'''Creates a new tensor with the same shape as `refTensor`, filled with a specified value.

    :param refTensor: Reference tensor used to determine the shape
    :type refTensor: tf.Tensor or np.ndarray
    :param fill_value: Value to fill the tensor with
    :return: A new tensor with the same shape as `refTensor`, filled with `fill_value`
    :rtype: np.ndarray or tf.Tensor
	'''
	shape = refTensor.shape
	if 'dtype' not in kwargs:
		kwargs['dtype'] = refTensor.dtype
	return full(shape, fill_value, refTensor, **kwargs)

def zeros(shape, refTensor, **kwargs):
	'''Creates a new tensor of a given shape, filled with :math:`0.0`.

    :param shape: Shape of the new tensor
    :type shape: int or list of int
    :param refTensor: Reference tensor that determines whether the output is a TensorFlow tensor or a NumPy ndarray.
    :type refTensor: tf.Tensor or np.ndarray
    :return: A new tensor, filled with `fill_value`
    :rtype: np.ndarray or tf.Tensor
	'''
	return full(shape, 0., refTensor, **kwargs)

def ones(shape, refTensor, **kwargs):
	'''Creates a new tensor of a given shape, filled with :math:`1.0`.

    :param shape: Shape of the new tensor
    :type shape: int or list of int
    :param refTensor: Reference tensor that determines whether the output is a TensorFlow tensor or a NumPy ndarray.
    :type refTensor: tf.Tensor or np.ndarray
    :return: A new tensor or array, filled with `fill_value`
    :rtype: np.ndarray or tf.Tensor
	'''
	return full(shape, 1., refTensor, **kwargs)


def copy(tensor, **kwargs):
	''' Creates a copy of the given tensor.

    :param tensor: The input tensor.
    :type tensor: tf.Tensor or np.ndarray
    :return: A copy of the input tensor.
    :rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.identity(tensor, **kwargs)
	else:
		return np.copy(tensor, **kwargs)


def stack(tensors, **kwargs):
	'''Packs a list of tensors along a new axis.

    :param tensors: A list or tuple of tensors to be stacked.
    :type tensors: list or tuple of np.ndarray or tf.Tensor
    :return: A stacked tensor or array with a new dimension.
    :rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(tensors[0], tf.Tensor):
		return tf.stack(tensors, **kwargs)
	else:
		return np.stack(tensors, **kwargs)


def castToComplex(tensor: tf.Tensor|np.ndarray):
	"""Casts a tensor to a complex128 type.

    :param tensor: The input tensor.
    :type tensor: tf.Tensor or np.ndarray
    :return: The input tensor cast to the complex128 data type.
    :rtype: tf.Tensor or np.ndarray
	"""
	if isinstance(tensor, tf.Tensor):
		return tf.cast(tensor, tf.complex128)
	elif isinstance(tensor, np.ndarray):
		return tensor.astype(np.complex128)
	else:
		return np.complex128(tensor)


def complex(real: tf.Tensor|np.ndarray, imag:tf.Tensor|np.ndarray ):
	'''Creates a complex number tensor from real and imaginary parts.

	:param real: Tensor representing the real part of a complex number.
	:type real: tf.Tensor or np.ndarray
	:param imag: Tensor representing the imaginary part of a complex number.
	:type imag: tf.Tensor or np.ndarray
	:return: A complex number tensor.
	:rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(real, tf.Tensor):
		return tf.complex(real, imag)
	else:
		return real + 1j*imag




def isComplex(tensor: tf.Tensor|np.ndarray) -> bool:
	'''Checks if a tensor is complex. Returns only single bool.

    :param tensor: The input tensor to check.
    :type tensor: tf.Tensor or np.ndarray
    :return: `True` if the tensor is complex, `False` otherwise.
    :rtype: bool
	'''
	if isinstance(tensor, tf.Tensor):
		return tensor.dtype in (tf.complex, tf.complex128, tf.complex64)
	else:
		return np.any(np.iscomplex(tensor))

def cross(a: tf.Tensor|np.ndarray, b: tf.Tensor|np.ndarray, axis: int = -1):
	"""Returns the cross product of `a` and `b`.

	Args:
		a (tf.Tensor | np.ndarray): Components of the first vector
		b (tf.Tensor | np.ndarray): Components of the last vector
		axis (int, optional): Axis of `a` and `b` along which the cross product is calculated for each entry in the other dimensions. Defaults to -1.

	Returns:
		tf.Tensor|np.ndarray: Cross product of `a` and `b`
	"""
	if isinstance(a, tf.Tensor):
		if axis == -1: # product along last axis
			return tf.linalg.cross(a, b)
		elif axis == 0: # product along first axis
			return tf.transpose(tf.linalg.cross(tf.transpose(a), tf.transpose(b)))
		else:
			logger.raiseException(Exception, f'Tensorflow cross product can be calculated only along the first `axis=0` or last `axis=-1` axis, but not along `axis={axis}`.')
	return np.cross(a, b, axis=axis)

def where(condition: tf.Tensor|np.ndarray, x: tf.Tensor|np.ndarray, y: tf.Tensor|np.ndarray):
	''' Selects elements from x or y based on the given condition.
    For each element:

    - If `condition` is zero the corresponding element from `x` is selected.
    - If `condition` is nonzero the corresponding element from `y` is selected.

    :param condition: A tensor determining which elements to select.
    :type condition: np.ndarray or tf.Tensor
    :param x: First input tensor.
    :type x: np.ndarray or tf.Tensor
    :param y: Second input tensor.
    :type y: np.ndarray or tf.Tensor
    :return: An tensor with elements selected from `x` or `y` based on `condition`.
    :rtype: np.ndarray or tf.Tensor
    '''
	if isinstance(condition, tf.Tensor):
		return tf.where(condition, x, y)
	else:
		return np.where(condition, x, y)

def zeros_like(tensor: tf.Tensor|np.ndarray):
	''' Returns a tensors with the same size as the input `tensor`. All elements are zero.

    :param tensor: Input tensor, the shape and type of `tensor` defines these same attributes of the returned tensor.
    :type tensor: np.ndarray or tf.Tensor
    :return: Tensor of zeros with the same shape and type as `tensor`.
    :rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.zeros_like(tensor)
	else:
		return np.zeros_like(tensor)

def ones_like(tensor: tf.Tensor|np.ndarray):
	''' Returns a tensors with the same size as the input `tensor`. All elements are one.

    :param tensor: Input tensor, the shape and type of `tensor` defines these same attributes of the returned tensor.
    :type tensor: np.ndarray or tf.Tensor
    :return: Tensor of ones with the same shape and type as `tensor`.
    :rtype: np.ndarray or tf.Tensor
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.ones_like(tensor)
	else:
		return np.ones_like(tensor)

def allclose(a: tf.Tensor|np.ndarray, b:tf.Tensor|np.ndarray, rtol=1e-05, atol=1e-08, equal_nan=False):
	'''
    Determines if two tensors are element-wise equal within a tolerance. See np.allclose.

    :param a: The first input tensor
    :type a: np.ndarray or tf.Tensor
    :param b: The second input tensor
    :type b: np.ndarray or tf.Tensor
    :param rtol: The relative tolerance, default is :math:`10^{-5}`
    :type rtol: float, optional
    :param atol: The absolute tolerance, default is :math:`10^{-8}`
    :type atol: float, optional
    :param equal_nan: Whether to treat NaN values as equal, default is False.
    :type equal_nan: bool, optional
    :return: `True` if the tensors are element-wise equal within the given tolerance, otherwise `False`.
    :rtype: bool or tf.Tensor
	'''
	if isinstance(a, tf.Tensor) or isinstance(b, tf.Tensor):
		return tf.experimental.numpy.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
	else:
		return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)

def tensordot(a: tf.Tensor|np.ndarray, b: tf.Tensor|np.ndarray, axes: int|tuple):
	'''
	Computes tensor dot product between `a` and `b` along the specified axes.

	:param a: First input tensor
	:type a: tf.Tensor | np.ndarray
	:param b: Second input tensor
	:type b: tf.Tensor | np.ndarray
	:param axes: The axis along which the dot product is computed. This can be:

		- An integer: If `axes` is an integer N, it sums over the last N axes of a and the first N axes of `b` in order
		- A tuple of two arrays: Specifies the axes of `a` and `b` to sum over. The first sequence applies to `a`, the second to `b`.
		- If axes=0, computes the outer product between `a` and `b`.
	:type axes: int | tuple
	:return: The result of the tensor dot product operation
	:rtype: tf.Tensor | np.ndarray
	'''

	if isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
		return tf.tensordot(a,b,axes)
	else:
		return np.tensordot(a,b,axes)

def arange(start, stop, reference_type: tf.Tensor|np.ndarray):
	''''Returns a tensor with values in a range between `start`and `stop`.

    :param start: The start value of the range. Defaults to :math:`0`.
    :param stop: The end value of the range (exclusive).
    :param reference_type: Reference tensor to determine the type of the result. If `reference_type` is a `tf.Tensor` a tf.Tensor is returned, otherwise a np.ndarray is returned.
	:type reference_type: tf.Tensor | np.ndarray
	:return: A tensor with values from the specified range.
	:rtype: tf.Tensor | np.ndarray
	'''
	if isinstance(reference_type, tf.Tensor):
		return tf.range(start, stop)
	else:
		return np.arange(start, stop)

def invert(tensor: tf.Tensor|np.ndarray):
	'''Computes the inverse of a tensor for all submatrices.

	:param tensor: Tensor to be inverted
	:type tensor: tf.Tensor | np.ndarray
	:return: The inverse of the input tensor, with the same type and shape as the input.
	:rtype: tf.Tensor | np.ndarray
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.linalg.inv(tensor)
	else:
		return np.linalg.inv(tensor)

def swapaxes(tensor: tf.Tensor|np.ndarray, axis1: int, axis2:int):
	'''Swaps two axes of a tensor.

	:param tensor: The input tensor.
	:type tensor: tf.Tensor or np.ndarray
	:param axis1: The first axis to swap.
	:type axis1: int
	:param axis2: The second axis to swap.
	:type axis2: int
	:return: A tensor with the specified axes swapped.
	:rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(tensor, tf.Tensor):
		rank = len(tensor.shape)
		perm = list(range(rank))
		perm[axis1], perm[axis2] = perm[axis2], perm[axis1]
		return tf.transpose(tensor, perm)
	return np.swapaxes(tensor, axis1, axis2)

def eye(N: int, refTensor: tf.Tensor|np.ndarray, dtype=float):
	'''Creates an identity tensor of size N x N.

	:param N: The size of the identity tensor.
	:type N: int
	:param refTensor: Reference tensor to determine the type of the result.
	:type refTensor: tf.Tensor | np.ndarray
	:return: An identity tensor of size N x N
	:rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(refTensor, tf.Tensor):
		return tf.eye(N, dtype=dtype)
	return np.eye(N, dtype=dtype)

def det(tensor: tf.Tensor|np.ndarray):
	'''Computes the determinant of a tensor.

	:param tensor: (..., M, M) Input array to compute determinants for.
	:type tensor: tf.Tensor | np.ndarray
	:return: The determinant of the input tensor.
	:rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.linalg.det(tensor)
	return np.linalg.det(tensor)

def diag(v: list|tf.Tensor|np.ndarray, refTensor: tf.Tensor|np.ndarray):
	'''Extract a diagonal or construct a diagonal tensor.

	:param v: If `v` is a 2-D array, return a copy of its `k`-th diagonal. If `v` is a 1-D array, return a 2-D array with `v` on the `k`-th diagonal.
	:type v: array_like
	:type refTensor: tf.Tensor | np.ndarray
	:return The extracted diagonal or constructed diagonal array.
	:rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(refTensor, tf.Tensor):
		return tf.linalg.diag(v)
	return np.diag(v)

def flip(tensor: tf.Tensor|np.ndarray, axis: int|tuple):
	'''Reverses a tensor along the specified axes.

	:param tensor: The input tensor.
	:type tensor: tf.Tensor or np.ndarray
	:param axis: Axis or axes along which to reverse the tensor.
	:type axis: int or tuple
	:return: A tensor with elements reversed along the given axes.
	:rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(tensor, tf.Tensor):
		return tf.reverse(tensor, axis=axis)
	return np.flip(tensor, axis=axis)


class Polynomial:
	'''
	Creates a polynomial class from a list of coefficients

	:param coefficients: Polynomial coefficient in order of increasing degrees, i.e. `(1,2,3)` corresponds to :math:`1 + 2x + 3x^2`
	:type coefficients: list or np.ndarray
	:param refTensor: Reference tensor to determine the type of the result. If `refTensor` is a `tf.Tensor` a tf.Tensor is returned, otherwise np.ndarray is returned.
	:type refTensor: tf.Tensor | np.ndarray

	:return: A function that evaluates the polynomial for a given input `x`
	'''

	def __init__(self, coefficients: list|np.ndarray, castComplex: bool = False, dtype: type = float):
		self.coefficients = coefficients
		self.castComplex = castComplex
		self.dtype = dtype


	@property
	def shape(self):
		'''Returns the number of coefficients of the polynomial.

		:return: The number of coefficients.
		:rtype: int
		'''
		return self.coefficients.shape[0]


	def __call__(self, x: tf.Tensor|np.ndarray):
		'''Evaluates the polynomial at the given input values.

		:param x: The input value(s) at which to evaluate the polynomial.
		:type x: tf.Tensor or np.ndarray
		:return: The polynomial evaluated at `x`.
		:rtype: tf.Tensor or np.ndarray
		'''
		if isinstance(x, tf.Tensor):
			# Highest-degree coefficient first for Horner's method

			reversedParameters = flip(self.coefficients, axis=[0])

			if self.castComplex:
				if not x.dtype in [tf.complex128, tf.complex64, complex]:
					x_eval = tf.complex(x, zeros_like(x))
				else:
					x_eval = x
				rev = tf.complex(reversedParameters, zeros_like(reversedParameters))
			else:
				x_eval = x
				rev = reversedParameters

			coeffs = tf.unstack(rev, axis=0)
			result = tf.zeros_like(x_eval, dtype=coeffs[0].dtype)
			for c in coeffs:
				result = result * x_eval + c
			return result
		else:
			polynomial = Poly(self.coefficients)
			return polynomial(x)
