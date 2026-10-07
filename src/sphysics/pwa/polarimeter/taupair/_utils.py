
# coding: utf-8
'''
Created on Friday 27 03 2026
Author: Stefan Wallner
Description: Utility functions for polarimeter frame rotations
'''

from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from .... import kinematics


def get_rotation_to_bodyfixed_nrk(p_tauMinus: np.ndarray, p_electron: np.ndarray) -> np.ndarray:
	'''Build rotation matrix to the nrk body-fixed frame.

	The coordinate convention follows
	Phys. Rev. D **109**, 032005 (2024), `doi: 10.1103/PhysRevD.109.032005 <https://doi.org/10.1103/PhysRevD.109.032005>`_

	:param p_tauMinus: Tau-minus momentum with shape ``(3, ...)`` or ``(4, ...)``
	:param p_electron: Electron momentum with shape ``(3, ...)`` or ``(4, ...)``
	:return: Rotation matrix with shape ``(3, 3, ...)``
	'''
	basis = kinematics.geometry.build_orthonormal_basis(p_tauMinus, p_electron)
	return kinematics.geometry.get_rotation_matrix_from_basis_vectors(*basis)


def rotate_to_bodyfixed_nrk(p_tauMinus: np.ndarray, p_electron: np.ndarray, *args: np.ndarray) -> tuple[np.ndarray, ...]:
	'''Rotate vectors to the nrk body-fixed frame.

	The coordinate convention follows
	Phys. Rev. D **109**, 032005 (2024), `doi: 10.1103/PhysRevD.109.032005 <https://doi.org/10.1103/PhysRevD.109.032005>`_

	:param p_tauMinus: Tau-minus momentum with shape ``(3, ...)`` or ``(4, ...)``
	:param p_electron: Electron momentum with shape ``(3, ...)`` or ``(4, ...)``
	:param args: Additional vectors, each with shape ``(3, ...)`` or ``(4, ...)``
	:return: Tuple of rotated vectors in the order ``(p_tauMinus, p_electron, *args)``
	'''
	rotation = get_rotation_to_bodyfixed_nrk(p_tauMinus, p_electron)
	rotated = []
	for p in (p_tauMinus, p_electron, *args):
		rotated.append(kinematics.geometry.apply_rotation_matrix(rotation, p))
	return tuple(rotated)
