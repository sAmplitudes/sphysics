# coding: utf-8
'''
Created on Wednesday 23 03 2022
@author: Stefan Wallner
@description: Additional math functions
'''

from __future__ import absolute_import, print_function, division, annotations

from typing import Any

import numpy as _np
import tensorflow as _tf

from ._tfnp import conjugate, real


def zeroBelowZero(value):
	''' Set every value below zero to zero.
    :param value: The input tensor
    :type value: tf.Tensor or np.ndarray
    :return: A tensor with zero if the input value is < :math:`0`.
    :rtype: tf.Tensor or np.ndarray
	'''
	if isinstance(value, _tf.Tensor):
		nonNegativeValue = _tf.cast(value > 0, value.dtype) * value
	elif isinstance(value, _np.ndarray):
		nonNegativeValue = value
		nonNegativeValue[nonNegativeValue < 0] = 0
	return nonNegativeValue


def sqrt(value, where=None, whereNotValue=None):
	'''	Calculates the square root of `value`.
    If `where` is given, calculate the square root only where `where` is True,
    all other elements will be set to `whereNotValue` (in contrast to numpy's where, where the values remain uninitialize).
    `whereNotValue` cannot be a tf.Tensor

    :param value: The input tensor
    :type value: tf.Tensor or np.ndarray
    :param where: A boolean tensor that indicates where to compute the square root. Defaults to `None`.
    :type where: tf.Tensor | np.ndarray
    :param whereNotValue: The value to assign when `where` is False. Defaults to `None`.
    :type whereNotValue: numeric
    :return: A tensor with the square roots computed if `where` is True, and a specified value `whereNotValue` if `where` is False.
    :rtype: tf.Tensor or np.ndarray
	'''
	#if where is not None:
	#	print(f"Debug: where={where}")

	if isinstance(value, _tf.Tensor):
		if where is not None:
			where = _tf.cast(where, value.dtype)
			valueMask = _tf.multiply(value, where)
			sqrtValue =_tf.math.sqrt(valueMask)
			sqrtValue = where*sqrtValue + (1.-where)*whereNotValue
		else:
			sqrtValue =_tf.math.sqrt(value)
	elif isinstance(value, _np.ndarray):
		if where is not None:
			sqrtValue = _np.empty_like(value)
			sqrtValue = _np.sqrt(value, where=where, out=sqrtValue)
			sqrtValue[~where] = whereNotValue                         # pylint: disable=invalid-unary-operand-type
		else:
			sqrtValue = _np.sqrt(value)
	else:
		if where is not None:
			sqrtValue = _np.sqrt(value) if where else whereNotValue
		else:
			sqrtValue = _np.sqrt(value)
	return sqrtValue


def sqrtZeroBelowZero(value):
	''' Computes the square root of value for `value` ≥ :math:`0`, else returns :math:`0.0`.

    :param value: The input tensor
    :type value: tf.Tensor or np.ndarray
    :return: A tensor with the square roots computed if the input value is ≥ :math:`0` and :math:`0.0` if the input value is < :math:`0`.
    :rtype: tf.Tensor or np.ndarray
	'''
	return sqrt(value, where=(value >= 0.), whereNotValue=0.0)


def abs2(tensor):
	''' Computes the absolute-squared value of the given tensor.

    :param tensor: The input tensor.
    :type tensor: tf.Tensor | np.ndarray
    :return: The absolute-squared value of the input tensor.
    :rtype: tf.Tensor | np.ndarray
	'''
	return real(tensor*conjugate(tensor))

def divide(numerator: _np.ndarray|_tf.Tensor, denominator: _np.ndarray|_tf.Tensor, where: _np.ndarray|_tf.Tensor|None=None, whereNotValue: Any=None) -> _np.ndarray|_tf.Tensor:
	''' Calculates the element-wise division of the numerator by the denominator.

    Calculate numerator / denominator.
    If `where` is given, calculate the fraction only where `where` is True,
    all other elements will be set to `whereNotValue` (in contrast to numpy's `where`, where the values remain uninitialize).
    `whereNotValue` cannot be a tf.Tensor

    :param numerator: The numerator of the division.
    :type numerator: _np.ndarray | _tf.Tensor
    :param denominator: The denominator of the division.
    :type denominator: _np.ndarray | _tf.Tensor
    :param where: A boolean tensor that determines where to compute the division.
                  The division is performed only where this value is True. Default is None.
    :type where: _np.ndarray | _tf.Tensor
    :param whereNotValue: The value to assign to elements where `where` is False. Cannot be a `tf.Tensor`.
    :type whereNotValue: Any, optional
    :return: The result of the division, with elements set to `whereNotValue` where `where` is False.
    :rtype: _np.ndarray | _tf.Tensor


	'''
	if isinstance(numerator, _tf.Tensor):
		if where is not None:
			where = _tf.cast(where, denominator.dtype)
			denominatorMask = _tf.multiply(denominator, where)
			fraction = _tf.math.divide_no_nan(numerator, denominatorMask)
			fraction = where*fraction + (1.-where)*whereNotValue
		else:
			fraction = numerator / denominator
	elif isinstance(numerator, _np.ndarray):
		if where is not None:
			where = _np.asarray(where, dtype=bool)
			fraction = _np.divide(numerator, denominator, where=where, out=None)
			fraction[~where] = whereNotValue
		else:
			fraction = _np.divide(numerator, denominator)
	else:
		if where is not None:
			fraction = _np.divide(numerator, denominator) if where else whereNotValue
		else:
			fraction = _np.divide(numerator, denominator)
	return fraction
