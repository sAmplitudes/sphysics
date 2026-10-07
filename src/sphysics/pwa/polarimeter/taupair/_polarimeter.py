# coding: utf-8
'''
Created on Friday 27 03 2026
Author: Stefan Wallner
Description: Tau polarimeter calculation utilities
'''

# pylint: disable=invalid-name

from __future__ import absolute_import, print_function, division, annotations

import numpy as np

from .... import math, lorentz


def get_polarimeter_from_J(J: np.ndarray, isParticle: np.ndarray, pTau: np.ndarray, pNu: np.ndarray) -> np.ndarray:
	'''Calculate the spin analyzer vector from the decay current.

	The polarimeter vector is defined in:
	Comput. Phys. Comm. **64**, 275-295 (1991), `doi: 10.1016/0010-4655(91)90038-M <https://doi.org/10.1016/0010-4655(91)90038-M>`_.
	"TAUOLA - a library of Monte Carlo programs to simulate decays of polarized τ leptons"
	and
	Comput. Phys. Comm. **245**, 109153 (2024), `doi: 10.1016/j.cpc.2024.109153 <https://doi.org/10.1016/j.cpc.2024.109153>`_.
	"The polarimeter vector for τ → 3π ν_τ decays"

	The current and momenta have to be given in the proper reference frame.
	The returned polarimeter vector is also defined in this frame.


	:param J: Decay current with shape ``(4, events)``
	:param isParticle: Boolean array indicating particle (True) vs antiparticle (False) with shape ``(events,)``
	:param pTau: Tau 4-momentum with shape ``(4, events)``
	:param pNu: Tau neutrino 4-momentum with shape ``(4, events)``
	:return: Spin analyzer vector (polarimeter vector) with shape ``(3, events)``
	'''
	va = math.full(pTau.shape[1], 1.0, pTau)
	va = math.where(isParticle, va, -1.0*va)

	mTau = math.sqrt(lorentz.lp(pTau, pTau))
	Jconj = math.conjugate(J)

	S = 2.0*( lorentz.lp(Jconj, pNu)*J + lorentz.lp(J, pNu)*Jconj - lorentz.lp(Jconj, J)*pNu )

	S5 = -2.0* math.imag(  math.einsum('mnrs,n,ne,r,re,s,se->me', lorentz.levitCivitaSymbol, lorentz.et, Jconj, lorentz.et, J, lorentz.et, pNu) )

	Stot = S5 - va*S

	w = lorentz.lp(pTau, S - va*S5)

	h = 1.0/(mTau*w) * (mTau**2 * Stot - lorentz.lp(pTau, Stot)*pTau)

	return math.real(h)
