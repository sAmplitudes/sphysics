# lorentz.py
'''
:Author: Fabian Krinner
:Description: Created on 2020-03-11
'''

from argparse import ArgumentError
import numpy as _np
from .utils import Logger as _Logger

from . import math as _math

_log = _Logger('lorentz')

##################
# Minkoski Metric
eta = _np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
			   [0., 0., -1., 0.], [0., 0., 0., -1.]])
# Diagonal of the metric to be used in einsum:
et = _np.array([eta[0, 0], eta[1, 1], eta[2, 2], eta[3, 3]])


##################
# Levi-Civita tensor
# Be aware: the Levi-Civita tensor is different from the Levi-Civita symbol by -1 in Minkowski space

levitCivitaSymbol = _np.zeros((4, 4, 4, 4))
levitCivitaSymbol[0, 1, 2, 3] = 1.
levitCivitaSymbol[0, 1, 3, 2] = -1.

levitCivitaSymbol[1, 0, 2, 3] = -1.
levitCivitaSymbol[1, 0, 3, 2] = 1.

levitCivitaSymbol[0, 2, 1, 3] = -1.
levitCivitaSymbol[0, 2, 3, 1] = 1.

levitCivitaSymbol[2, 0, 1, 3] = 1.
levitCivitaSymbol[2, 0, 3, 1] = -1.

levitCivitaSymbol[0, 3, 1, 2] = 1.
levitCivitaSymbol[0, 3, 2, 1] = -1.

levitCivitaSymbol[3, 0, 1, 2] = -1.
levitCivitaSymbol[3, 0, 2, 1] = 1.

levitCivitaSymbol[1, 2, 0, 3] = 1.
levitCivitaSymbol[1, 2, 3, 0] = -1.

levitCivitaSymbol[2, 1, 0, 3] = -1.
levitCivitaSymbol[2, 1, 3, 0] = 1.

levitCivitaSymbol[1, 3, 0, 2] = -1.
levitCivitaSymbol[1, 3, 2, 0] = 1.

levitCivitaSymbol[3, 1, 0, 2] = 1.
levitCivitaSymbol[3, 1, 2, 0] = -1.

levitCivitaSymbol[2, 3, 1, 0] = -1.
levitCivitaSymbol[2, 3, 0, 1] = 1.

levitCivitaSymbol[3, 2, 1, 0] = 1.
levitCivitaSymbol[3, 2, 0, 1] = -1.

epsilon = levitCivitaTensor = -1.*levitCivitaSymbol


def lp(a, b):                                         #pylint: disable=invalid-name
	"""Lorentz product of lorentz vectors a and b.
	The contraction is performed along the first axis of a and b
	"""
	return _math.einsum('i...,i...,i->...', a, b, et)


def tensorproduct(*args):
	'''
	Calculates the outer tensor product of a, b. ....
	The last index of a, b, ... is assumed to represent by the event index
	and is the same for all input tensors and the output tensor, i.e.
	rank(output) :math:`= \\sum_i` rank :math:`(a_i) - 1`
	'''
	if len(args) < 1:
		_log.raiseException(ArgumentError, "`tensorproduct` requires at least one arguments!")
	if len(args) > 25 :
		_log.raiseException(ArgumentError, "`tensorproduct` can handle only up to 25 arguments!")

	expressions = []
	iStart = ord('a')
	for tensor in args:
		aIndices = ''
		iEnd = iStart+len(tensor.shape)-1
		for i in range(iStart, iEnd):
			aIndices+=chr(i)
		expressions.append(aIndices)
		iStart = iEnd
	inputExpression = ','.join([ f'{expression}z' for expression in expressions])
	outputExpression = ''.join(expressions) + 'z' # z is the event index
	return _math.einsum(f'{inputExpression}->{outputExpression}', *args)


def getBoostFromRestFrame(p):
	'''
	Boost matrices in the direction of p, i.e. boosts from the p rest frame to the frame where p is defined

	:param p: numpy array of 4-vectors of the shape = (4, nEvents)
	'''
	return _getBoost(p, toSystem=True)


def getBoostToRestFrame(p):
	'''
	Boost matrices in the inverse direction of p, i.e. boosts from the frame where p is defined to the p rest frame

	:param p: numpy array of 4-vectors of the shape = (4, nEvents)
	'''
	return _getBoost(p, toSystem=False)


def _getBoost(p, toSystem):
	"""
	Returns an npmuy array of boosts with shape (4,4,nEvents)
	If toSystem is True, the returned array boosts from the p rest frame to the frame where p is defined
	If toSystem is False, the returned array boosts from the frame where p is defined into the p rest frame

	:param p: numpy array of 4-vectors of the shape = (4, nEvents)
	:param toSystem: flag, whether to boost to or from the system specified by p
	"""
	boostMatrix = _np.empty((4,4,p.shape[-1]))

	absP2 = _math.sum(p[1:4]**2, axis = 0)
	E     = p[0]
	m     = _math.sqrt(E**2 - absP2)
	absP  = _math.sqrt(absP2)
	gamma = E/m
	beta  = absP/E

	zeros  = absP2 == 0.

	direction = _math.full_like(p, fill_value=_np.nan)
	direction[:,~zeros] = p[:,~zeros]/absP[None,~zeros]
	direction[0] = 1.

	sign = 1. if toSystem else -1.
	betaGamma = sign * beta * gamma

	boostMatrix[:,:,:] = gamma[None,None,:] - 1.
	boostMatrix[:,0,:] = betaGamma[None,:]
	boostMatrix[0,:,:] = betaGamma[None,:]
	boostMatrix *= direction[None,:,:]*direction[:,None,:]
	boostMatrix[:] += _np.identity(4).reshape(4,4,1)
	boostMatrix[0,0,:] = gamma

	boostMatrix[...,zeros] = _np.identity(4).reshape(4,4,1)

	return boostMatrix

def applyBoost(boost: _np.ndarray, p: _np.ndarray) -> _np.ndarray:
	"""Apply boost matrix to lorentz vector

	Args:
		boost (_np.ndarray): Boost matrix of shape (4,4,.....)
		p (_np.ndarray): Lorentz vector to be boosted by the boost matrix of shape (4, ....)

	Returns:
		_np.ndarray: Boosted lorentz vector
	"""
	return _math.einsum('ij...,j...->i...', boost, p)
