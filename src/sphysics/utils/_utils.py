# coding: utf-8
'''
:Author: Stefan Wallner
'''

import numpy as np
import polars as pl

import awkward as ak
import uproot


def pdg_rounding(value: float, uncertainty: float) -> tuple[float, float, str, str]:
	'''
	Round a value and its uncertainty following the PDG rounding rules.

	The number of significant digits kept for the uncertainty is determined from its three highest-order
	digits: if they lie between 100 and 354, the uncertainty is rounded to two significant digits; if they
	lie between 355 and 949, it is rounded to one significant digit; if they lie between 950 and 999, the
	uncertainty is rounded up to 1000, which is equivalent to two significant digits at the next power of
	ten. `value` is rounded to the same decimal place as the uncertainty.

	:param value: Central value to round
	:param uncertainty: Uncertainty of `value`, used to determine the number of significant digits kept
	:return: Tuple `(value_rounded, uncertainty_rounded, value_str, uncertainty_str)` with the rounded value
		and uncertainty as floats, and their string representations with the appropriate number of decimal places
	'''
	digits = int(np.floor(np.log10(uncertainty)-2))
	uncertainty_scaled = np.floor(uncertainty/10**digits)

	if uncertainty_scaled <= 354:
		digits += 1
	elif uncertainty_scaled <= 949:
		digits += 2
	else:
		digits += 2

	digits = -digits

	value_rounded, uncertainty_rounded = float(np.round(value, digits)), float(np.round(uncertainty, digits))

	digits = max(digits, 0)
	return value_rounded, uncertainty_rounded, f'{value_rounded:.{digits}f}', f'{uncertainty_rounded:.{digits}f}'



def load_root_tree_to_polars(file_path: str, tree_name: str, branches: list[str]|None = None) -> pl.DataFrame:
	'''
	Load a ROOT TTree into a polars DataFrame.

	The tree is read with `uproot` into an awkward array, which is then handed to polars through an
	Arrow table. Awkward arrays, Arrow and polars all store a column as contiguous data and offset
	buffers in the same memory layout, so this conversion needs no per-entry Python objects and the
	returned DataFrame shares its buffers with the data read from the file. Branches of a simple,
	flat type become scalar columns, jagged branches of variable length become `List` columns.

	Reading the whole tree still needs memory for the full result, so for a tree that does not fit
	into memory pass `branches` to read only the columns that are actually needed.

	:param file_path: Path to the ROOT file
	:param tree_name: Name of the TTree inside the ROOT file
	:param branches: Names of the branches to read, `None` reads all branches of the tree
	:return: Tree data as a polars DataFrame, with jagged branches stored as `List` columns
	'''
	with uproot.open(file_path) as f:
		tree = f[tree_name]
		arrays = tree.arrays(branches, library='ak')

	# `extensionarray = False` strips the awkward-specific Arrow extension type, which polars cannot
	# interpret and would otherwise only load as its storage type, emitting a warning
	return pl.from_arrow(ak.to_arrow_table(arrays, extensionarray=False))
