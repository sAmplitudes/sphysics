# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Geometric utilities for kinematic vector transformations
'''

# pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import numpy as np
from .. import math
from ..utils import Logger

log = Logger('kinematics')

def build_orthonormal_basis( p: np.ndarray, q: np.ndarray ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	'''Build an orthonormal basis from two input vectors.

	(u, v, w) form a right-handed orthonormal basis where w is aligned with p,
	v is in the plane spanned by p and q, and u is perpendicular to that plane.

	:param p: Input vector with shape ``(3, ...)`` or ``(4, ...)``
	:param q: Input vector with shape ``(3, ...)`` or ``(4, ...)``
	:return: Tuple ``(u, v, w)`` of orthonormal basis vectors, each with shape ``(3, ...)``
	'''
	if p.shape[0] not in (3,4) or q.shape[0] not in (3,4):
		log.raiseException(ValueError, f'Expected p and q to be 3- or 4-vectors, got shapes {p.shape} and {q.shape}')

	if p.shape[0] == 4:
		p = p[1:]
	if q.shape[0] == 4:
		q = q[1:]

	q_unit = q / math.sqrt(math.sum(q**2,axis=0))

	w = p / math.sqrt(math.sum(p**2,axis=0))

	cosTheta = math.sum(q_unit * w, axis=0)
	sinTheta = math.sqrt(1.0 - cosTheta**2)

	u = math.cross(q_unit,w, axis=0)/sinTheta

	# the same as w x u
	v = (q_unit - cosTheta*w)/sinTheta

	return (u, v, w)


def get_rotation_matrix_from_basis_vectors( u: np.ndarray, v: np.ndarray, w: np.ndarray ) -> np.ndarray:
	'''Build a rotation matrix from orthonormal basis vectors.

	:param u: First basis vector with expected shape ``(3, ...)``
	:param v: Second basis vector with expected shape ``(3, ...)``
	:param w: Third basis vector with expected shape ``(3, ...)``
	:return: Rotation matrix with shape ``(3, 3, ...)``
	'''
	return math.stack((u, v, w), axis=0)


def apply_rotation_matrix( R: np.ndarray, p: np.ndarray ) -> np.ndarray:
	'''Apply a rotation matrix to 3- or 4-vectors.

	:param R: Rotation matrix with shape ``(3, 3, ...)``
	:param p: Input vector with shape ``(3, ...)`` or ``(4, ...)``
	:return: Rotated vector with the same shape as ``p``
	'''
	if p.shape[0] == 3:
		return  math.einsum('ij...,j...->i...', R, p)
	if p.shape[0] == 4:
		p = math.copy(p)
		p[1:] = apply_rotation_matrix(R, p[1:])
		return p
	log.raiseException(ValueError, f'Expected p to be a 3- or 4-vector, got shape {p.shape}')
	return None
