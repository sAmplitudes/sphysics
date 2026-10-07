#!/usr/bin/env python
# coding: utf-8
'''


@author: Stefan Wallner
'''

from __future__ import absolute_import, print_function, division

import unittest
import scipy.integrate
import pandas as pd

import sphysics
from sphysics import Constants as C

import numpy as np
import tensorflow as tf



class KinematicsTest(unittest.TestCase):

	def __init__(self, methodName: str = ...) -> None:
		super().__init__(methodName)


	def test_fourbodyPhasespaceInt12(self):
		m1 = C.M.pi0
		m2 = C.M.pi
		m3 = C.M.K
		m4 = 0.

		for n in [10, 100, 1_000]:
			rows = []
			for s1234 in [C.M.tau**2, C.M.B0**2, 40**0]:
				for m123 in np.linspace(m1+m2+m3, s1234**0.5-m4, 20):
					intGaus = sphysics.kinematics.fourbodyPhasespaceInt12(s1234, m123, m1, m2, m3, m4, nPoints=n)
					intQuad, errQuad = scipy.integrate.quad_vec(lambda m12: sphysics.kinematics.fourbodyPhasespace(s1234, m123, m12, m1, m2, m3, m4), m1+m2, m123-m3)
					if errQuad == 0:
						errQuad = 1
					delta = (intGaus - intQuad)
					delta_rel = delta/(intQuad if intQuad != 0. else 1.)
					pull = delta/errQuad
					rows.append([s1234, m123, pull[0], delta[0], delta_rel[0], intGaus, intQuad, errQuad])
			df = pd.DataFrame(rows, columns=['s1234', 'm123', 'pull', 'delta', 'delta rel.', 'intGaus', 'intQuad', 'errquad'])

			limit = {10: 1e-3, 100: 1e-6, 1000: 1e-9}
			self.assertLess(df["delta rel."].abs().max(), limit[n])





if __name__ == '__main__':
	sphysics.random.generators.setSeed(1234)
	unittest.main()
