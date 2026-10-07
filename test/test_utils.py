#!/usr/bin/env python
# coding: utf-8
'''


@author: Stefan Wallner
'''

from __future__ import absolute_import, print_function, division

import os
import tempfile
import unittest

import numpy as np
import polars as pl

import awkward as ak
import uproot

import sphysics


class UtilsTest(unittest.TestCase):
	def test_pdg_rounding_two_significant_digits(self):
		# uncertainty's three highest-order digits in [100, 354] -> round to two significant digits
		value, uncertainty, value_str, uncertainty_str = sphysics.utils.pdg_rounding(1.2345, 0.0234)
		self.assertAlmostEqual(value, 1.234, 6)
		self.assertAlmostEqual(uncertainty, 0.023, 6)
		self.assertEqual(value_str, '1.234')
		self.assertEqual(uncertainty_str, '0.023')

	def test_pdg_rounding_one_significant_digit(self):
		# uncertainty's three highest-order digits in [355, 949] -> round to one significant digit
		value, uncertainty, value_str, uncertainty_str = sphysics.utils.pdg_rounding(1.2345, 0.05)
		self.assertAlmostEqual(value, 1.23, 6)
		self.assertAlmostEqual(uncertainty, 0.05, 6)
		self.assertEqual(value_str, '1.23')
		self.assertEqual(uncertainty_str, '0.05')

	def test_pdg_rounding_round_up_to_next_power_of_ten(self):
		# uncertainty's three highest-order digits in [950, 999] -> round up to two significant digits
		# at the next power of ten
		value, uncertainty, value_str, uncertainty_str = sphysics.utils.pdg_rounding(0.827, 0.0958)
		self.assertAlmostEqual(value, 0.83, 6)
		self.assertAlmostEqual(uncertainty, 0.1, 6)
		self.assertEqual(value_str, '0.83')
		self.assertEqual(uncertainty_str, '0.10')

	def test_pdg_rounding_uncertainty_larger_than_one(self):
		value, uncertainty, value_str, uncertainty_str = sphysics.utils.pdg_rounding(1234.5, 234)
		self.assertAlmostEqual(value, 1230., 6)
		self.assertAlmostEqual(uncertainty, 230., 6)
		self.assertEqual(value_str, '1230')
		self.assertEqual(uncertainty_str, '230')


class LoadRootTreeToPolarsTest(unittest.TestCase):
	'''
	Tests of `sphysics.utils.load_root_tree_to_polars`, run against a small TTree written on the fly
	that holds one branch of each kind the conversion has to handle.
	'''

	FLAT_FLOAT = [1.5, -2.5, 3.5, 0.0, 7.25]
	FLAT_INT   = [1, 2, 3, 4, 5]
	FLAT_BOOL  = [True, False, True, True, False]
	# a jagged branch with a varying number of entries per event, including an empty one
	JAGGED     = [[1.0, 2.0], [], [3.0], [4.0, 5.0, 6.0], [7.0]]

	def setUp(self):
		self._tmpDir = tempfile.TemporaryDirectory()
		self.filePath = os.path.join(self._tmpDir.name, 'tree.root')
		with uproot.recreate(self.filePath) as outFile:
			outFile['tree'] = {'flat_float': np.array(self.FLAT_FLOAT, dtype=np.float64),
			                   'flat_int'  : np.array(self.FLAT_INT, dtype=np.int32),
			                   'flat_bool' : np.array(self.FLAT_BOOL, dtype=bool),
			                   'jagged'    : ak.Array(self.JAGGED)}

	def tearDown(self):
		self._tmpDir.cleanup()

	def test_all_branches_are_read_by_default(self):
		dataFrame = sphysics.utils.load_root_tree_to_polars(self.filePath, 'tree')
		self.assertIsInstance(dataFrame, pl.DataFrame)
		self.assertEqual(dataFrame.height, len(self.FLAT_INT))
		self.assertEqual(sorted(dataFrame.columns), ['flat_bool', 'flat_float', 'flat_int', 'jagged'])

	def test_flat_branch_values(self):
		dataFrame = sphysics.utils.load_root_tree_to_polars(self.filePath, 'tree')
		self.assertEqual(dataFrame['flat_float'].to_list(), self.FLAT_FLOAT)
		self.assertEqual(dataFrame['flat_int'].to_list(), self.FLAT_INT)
		self.assertEqual(dataFrame['flat_bool'].to_list(), self.FLAT_BOOL)

	def test_flat_branch_types_are_preserved(self):
		# the branch types must survive the conversion, in particular the integer branch must not be
		# widened to a float column
		dataFrame = sphysics.utils.load_root_tree_to_polars(self.filePath, 'tree')
		self.assertEqual(dataFrame.schema['flat_float'], pl.Float64)
		self.assertEqual(dataFrame.schema['flat_int'], pl.Int32)
		self.assertEqual(dataFrame.schema['flat_bool'], pl.Boolean)

	def test_jagged_branch_becomes_list_column(self):
		# a jagged branch has to keep its variable length per event, i.e. it may neither be flattened
		# into a scalar column nor padded to a fixed length
		dataFrame = sphysics.utils.load_root_tree_to_polars(self.filePath, 'tree')
		self.assertEqual(dataFrame.schema['jagged'], pl.List(pl.Float64))
		self.assertEqual(dataFrame['jagged'].to_list(), self.JAGGED)

	def test_branches_argument_selects_subset(self):
		dataFrame = sphysics.utils.load_root_tree_to_polars(self.filePath, 'tree',
		                                                    branches=['flat_float', 'jagged'])
		self.assertEqual(sorted(dataFrame.columns), ['flat_float', 'jagged'])
		self.assertEqual(dataFrame.height, len(self.FLAT_INT))
		self.assertEqual(dataFrame['flat_float'].to_list(), self.FLAT_FLOAT)
		self.assertEqual(dataFrame['jagged'].to_list(), self.JAGGED)

	def test_empty_tree_keeps_its_schema(self):
		# a tree without any entry must give an empty DataFrame of the correct schema, rather than
		# failing or losing the list type of the jagged branch
		filePath = os.path.join(self._tmpDir.name, 'empty.root')
		with uproot.recreate(filePath) as outFile:
			outFile['tree'] = {'flat_float': np.array([], dtype=np.float64),
			                   'jagged'    : ak.Array([[1.0]])[:0]}

		dataFrame = sphysics.utils.load_root_tree_to_polars(filePath, 'tree')
		self.assertEqual(dataFrame.height, 0)
		self.assertEqual(dataFrame.schema['flat_float'], pl.Float64)
		self.assertEqual(dataFrame.schema['jagged'], pl.List(pl.Float64))

	def test_unknown_tree_name_raises(self):
		with self.assertRaises(KeyError):
			sphysics.utils.load_root_tree_to_polars(self.filePath, 'not_a_tree')


if __name__ == '__main__':
	unittest.main()
