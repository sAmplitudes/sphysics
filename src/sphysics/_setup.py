# coding: utf-8
'''
Created on Apr 24, 2020

@author: stefan
'''
#pylint: disable=unused-import,wrong-import-position

from __future__ import absolute_import, print_function, division

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

 # has to be imported before ROOT :(
import iminuit as _iminuit # pylint: disable=wrong-import-order

# import ROOT if installed to preserve import order
try:
	import ROOT
except ImportError:
	pass


# Include necessary libraries
from . import _init_cpplibs


from .utils import Logger


from ._core.utilities import Environment  # pylint: disable=wrong-import-position,import-error

_log = Logger("sphysics")
if Environment().isGitStatusClean():
	_log.info(Environment().getSummary())
else:
	_log.warning(Environment().getSummary())

from ._root_path import root_path

os.environ['SPHYSICS'] = root_path
if  'SPHYSICS_PARTICLEDATATABLE' not in os.environ: # set default particle datatable
	os.environ['SPHYSICS_PARTICLEDATATABLE'] = root_path + '/particleData/particleDataTable2022.txt'
	_log.warning(f"Environment variable `SPHYSICS_PARTICLEDATATABLE` not set. Using default `{os.path.basename(os.environ['SPHYSICS_PARTICLEDATATABLE'])}`.")
