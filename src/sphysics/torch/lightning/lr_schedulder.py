# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Mixins for learning-rate schedulers of Lightning modules
'''

#pylint: disable=unused-import

from __future__ import absolute_import

from ._lr_scheduler import ReduceOnPlateauMixin, ExponentialLRMixin
