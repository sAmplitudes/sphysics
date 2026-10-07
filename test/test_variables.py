#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from sphysics.eventselection._variables import Variables, VariablesBase


class VariablesTest(unittest.TestCase):
	def test_variables_base_registration_and_consistency(self):
		variables = VariablesBase()
		self.assertTrue(variables.addVariable('values'))
		self.assertFalse(variables.addVariable('values'))
		variables['values'] = np.array([1.0, 2.0])
		variables.addVariable('scale')
		variables['scale'] = 3.0
		variables.addVariable('metadata')
		variables['metadata'] = {'source': 'test'}

		self.assertEqual(variables.getVariables(), ['metadata', 'scale', 'values'])
		self.assertEqual(variables.getLoadedNonscalarVariables(), ['values'])
		self.assertEqual(variables.getLoadedScalarVariables(), ['metadata', 'scale'])
		self.assertEqual(variables.nEvents, 2)
		self.assertTrue(variables.check())

		variables.addVariable('mismatched')
		variables['mismatched'] = np.array([1.0, 2.0, 3.0])
		with self.assertRaises(Exception):
			variables.check()

	def test_variables_base_polars_conversion_consumes_data(self):
		variables = VariablesBase()
		variables.addVariable('mass')
		variables['mass'] = np.array([1.0, 2.0])
		variables.addVariable('scale')
		variables['scale'] = 3.0
		variables.addVariable('angles')
		variables['angles'] = np.array([[0.1, 0.2], [0.3, 0.4]])

		dataFrame = variables.toPolarsDataFrame()
		restored = VariablesBase.fromPolarsDataFrame(dataFrame)

		self.assertEqual(dataFrame.columns, [])
		self.assertIsInstance(restored, VariablesBase)
		self.assertIsNone(variables['mass'])
		self.assertIsNone(variables['scale'])
		self.assertIsNone(variables['angles'])
		np.testing.assert_array_equal(restored['mass'], [1.0, 2.0])
		np.testing.assert_array_equal(restored['scale'], [3.0, 3.0])
		np.testing.assert_array_equal(restored['angles[0]'], [0.1, 0.2])
		np.testing.assert_array_equal(restored['angles[1]'], [0.3, 0.4])

	def test_copy_and_empty_copy_preserve_expected_structure(self):
		variables = Variables()
		variables.addVariable('p', branchname='track_p', auto=False, is4Momentum=True)
		variables['p'] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
		variables.addVariable('scale')
		variables['scale'] = 2.0

		copied = variables.copy()
		empty = variables.emptyCopy()
		copied['p'][0, 0] = 99.0

		self.assertEqual(variables['p'][0, 0], 1.0)
		self.assertEqual(copied.getBranchname('p'), 'track_p')
		self.assertFalse(copied.getAuto('p'))
		self.assertTrue(empty.getIs4Momentum('p'))
		self.assertEqual(empty.get4MomentumVars('p'), ['E', 'px', 'py', 'pz'])
		self.assertIsNone(empty['p'])
		self.assertIsNone(empty['scale'])

	def test_iadd_offsets_channel_indices_and_can_consume_other(self):
		first = Variables()
		first.addVariable('mass')
		first['mass'] = np.array([1.0, 2.0])
		first.addVariable('scale')
		first['scale'] = 5.0
		first.setChannelIndexMap('signal', np.array([0, 1]))
		second = Variables()
		second.addVariable('mass')
		second['mass'] = np.array([3.0])
		second.addVariable('scale')
		second['scale'] = 5.0
		second.setChannelIndexMap('signal', np.array([0]))

		first.iadd(second, moveOther=True)

		np.testing.assert_array_equal(first['mass'], [1.0, 2.0, 3.0])
		np.testing.assert_array_equal(first.idx('signal'), [0, 1, 2])
		self.assertFalse(second.hasVariable('mass'))
		self.assertEqual(first.nEvents, 3)

	def test_iadd_can_consume_zero_copy_polars_data(self):
		first = Variables()
		first.addVariable('mass')
		first['mass'] = np.array([1.0, 2.0])
		second = Variables.fromPolarsDataFrame(pl.DataFrame({'mass': [3.0, 4.0]}))

		first.iadd(second, moveOther=True)

		np.testing.assert_array_equal(first['mass'], [1.0, 2.0, 3.0, 4.0])
		self.assertFalse(second.hasVariable('mass'))

	def test_filter_remaps_channel_indices_without_modifying_source(self):
		variables = Variables()
		variables.addVariable('mass')
		variables['mass'] = np.array([1.0, 2.0, 3.0, 4.0])
		variables.addVariable('p', is4Momentum=True)
		variables['p'] = np.arange(16.0).reshape(4, 4)
		variables.setChannelIndexMap('signal', np.array([0, 2]))
		variables.setChannelIndexMap('background', np.array([1, 3]))

		filtered = variables.filter(np.array([True, False, True, False]))

		np.testing.assert_array_equal(filtered['mass'], [1.0, 3.0])
		np.testing.assert_array_equal(filtered['p'], [[0.0, 2.0], [4.0, 6.0], [8.0, 10.0], [12.0, 14.0]])
		np.testing.assert_array_equal(filtered.idx('signal'), [0, 1])
		np.testing.assert_array_equal(filtered.idx('background'), [])
		np.testing.assert_array_equal(variables['mass'], [1.0, 2.0, 3.0, 4.0])

	def test_channels_weights_merging_order_and_labels(self):
		variables = Variables()
		variables.addVariable('weight')
		variables['weight'] = np.array([1.0, 2.0, 3.0, 4.0])
		variables.setChannelIndexMap('signal', np.array([0, 2]))
		variables.setChannelIndexMap('background', np.array([1, 3]))
		variables.setChannelOrder(['signal', 'background'])
		variables.setChannelLabels({'signal': 'Signal'})

		self.assertEqual(variables.getChannels(), ['signal', 'background'])
		self.assertEqual(variables.getNevents(), {'signal': 2, 'background': 2})
		self.assertEqual(variables.getWeightedNevents(), {'signal': 4.0, 'background': 6.0})
		self.assertEqual(variables.getChannelLabel('signal'), 'Signal')
		self.assertEqual(variables.getChannelLabel('background'), 'background')

		variables.mergeChannels({'all': ['signal', 'background']})

		self.assertEqual(variables.getChannels(), ['all'])
		np.testing.assert_array_equal(variables.idx('all'), [0, 1, 2, 3])

	def test_pandas_conversion_flattens_two_dimensional_arrays(self):
		variables = Variables()
		variables.addVariable('mass')
		variables['mass'] = np.array([1.0, 2.0])
		variables.addVariable('angles')
		variables['angles'] = np.array([[0.1, 0.2], [0.3, 0.4]])

		dataFrame = variables.toDataFrame()
		restored = Variables.fromDataFrame(dataFrame)

		self.assertEqual(list(dataFrame.columns), ['mass', 'angles[0]', 'angles[1]'])
		np.testing.assert_array_equal(restored['mass'], [1.0, 2.0])
		np.testing.assert_array_equal(restored['angles[0]'], [0.1, 0.2])
		np.testing.assert_array_equal(restored['angles[1]'], [0.3, 0.4])

	def test_split_random_partitions_every_event(self):
		variables = VariablesBase()
		variables.addVariable('event')
		variables['event'] = np.arange(10)

		subsets = variables.splitRandom([0.3, 0.2])

		self.assertEqual([subset.nEvents for subset in subsets], [3, 2, 5])
		np.testing.assert_array_equal(np.sort(np.hstack([subset['event'] for subset in subsets])), np.arange(10))

	def test_polars_round_trip_consumes_data(self):
		variables = Variables()
		variables.addVariable('mass')
		mass = np.array([1.0, 2.0, 3.0])
		variables['mass'] = mass
		variables.addVariable('scale')
		variables['scale'] = 2.0
		variables.addVariable('p', is4Momentum=True)
		variables['p'] = np.array([
			[10.0, 20.0, 30.0],
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])

		dataFrame = variables.toPolarsDataFrame()

		self.assertEqual(dataFrame.columns, ['mass', 'p_E', 'p_px', 'p_py', 'p_pz', 'scale'])
		self.assertTrue(np.shares_memory(mass, dataFrame['mass'].to_numpy(allow_copy=False)))
		self.assertIsNone(variables['mass'])
		self.assertIsNone(variables['p'])
		self.assertIsNone(variables['scale'])
		restored = Variables.fromPolarsDataFrame(dataFrame)
		self.assertEqual(dataFrame.columns, [])
		self.assertTrue(restored.getIs4Momentum('p'))
		np.testing.assert_array_equal(restored['mass'], [1.0, 2.0, 3.0])
		np.testing.assert_array_equal(restored['scale'], [2.0, 2.0, 2.0])
		np.testing.assert_array_equal(restored['p'], [
			[10.0, 20.0, 30.0],
			[1.0, 2.0, 3.0],
			[4.0, 5.0, 6.0],
			[7.0, 8.0, 9.0],
		])

	def test_incomplete_momentum_columns_remain_variables(self):
		dataFrame = pl.DataFrame({
			'p_E': [10.0, 20.0],
			'p_px': [1.0, 2.0],
			'p_py': [3.0, 4.0],
		})

		variables = Variables.fromPolarsDataFrame(dataFrame)

		self.assertEqual(dataFrame.columns, [])
		self.assertFalse(variables.getIs4Momentum('p_E'))
		self.assertFalse(variables.getIs4Momentum('p_px'))
		self.assertFalse(variables.getIs4Momentum('p_py'))
		np.testing.assert_array_equal(variables['p_E'], [10.0, 20.0])

	def test_from_polars_shares_buffers_of_one_dimensional_variables(self):
		values = np.arange(4, dtype=np.float64)
		dataFrame = pl.DataFrame([pl.Series('mass', values)])

		variables = Variables.fromPolarsDataFrame(dataFrame)

		self.assertTrue(np.shares_memory(values, variables['mass']))
		self.assertFalse(variables['mass'].flags.writeable)

		dataFrame = pl.DataFrame([pl.Series('mass', values)])
		copied = Variables.fromPolarsDataFrame(dataFrame, zeroCopy=False)
		self.assertFalse(np.shares_memory(values, copied['mass']))
		self.assertTrue(copied['mass'].flags.writeable)

	def test_from_polars_handles_columns_without_zero_copy_layout(self):
		dataFrame = pl.DataFrame({'flag': [True, False, True], 'value': [1.0, None, 3.0]})

		variables = Variables.fromPolarsDataFrame(dataFrame)

		self.assertEqual(dataFrame.columns, [])
		np.testing.assert_array_equal(variables['flag'], [True, False, True])
		np.testing.assert_array_equal(variables['value'], [1.0, np.nan, 3.0])

	def test_from_polars_preserves_momentum_dtype(self):
		dataFrame = pl.DataFrame({
			f'p_{component}': np.arange(3, dtype=np.float32) for component in ['E', 'px', 'py', 'pz']
		})

		variables = Variables.fromPolarsDataFrame(dataFrame)

		self.assertEqual(variables['p'].dtype, np.float32)

	def test_load_parquet_excludes_variables_before_reading_columns(self):
		with tempfile.TemporaryDirectory() as temporaryDirectory:
			path = Path(temporaryDirectory) / 'variables.parquet'
			variables = Variables()
			variables.addVariable('mass')
			variables['mass'] = np.array([1.0, 2.0])
			variables.addVariable('weight')
			variables['weight'] = np.array([3.0, 4.0])
			variables.addVariable('p', is4Momentum=True)
			variables['p'] = np.arange(8.0).reshape(4, 2)
			variables.store(str(path))

			loaded = Variables.load(str(path), printInfo=False, excludeVariables=['mass', 'p'])

			self.assertFalse(loaded.hasVariable('mass'))
			self.assertFalse(loaded.hasVariable('p'))
			np.testing.assert_array_equal(loaded['weight'], [3.0, 4.0])

	def test_two_dimensional_variable_uses_indexed_columns(self):
		variables = Variables()
		variables.addVariable('angles')
		variables['angles'] = np.array([
			[0.1, 0.2],
			[0.3, 0.4],
		])

		dataFrame = variables.toPolarsDataFrame()

		self.assertEqual(dataFrame.columns, ['angles[0]', 'angles[1]'])
		self.assertIsNone(variables['angles'])
		np.testing.assert_array_equal(dataFrame['angles[0]'].to_numpy(allow_copy=False), [0.1, 0.2])
		np.testing.assert_array_equal(dataFrame['angles[1]'].to_numpy(allow_copy=False), [0.3, 0.4])


if __name__ == '__main__':
	unittest.main()
