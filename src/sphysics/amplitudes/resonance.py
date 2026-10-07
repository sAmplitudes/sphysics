# coding: utf-8
'''
:Author: Stefan Wallner
:Description: Created on Nov 7, 2019
'''

# pylint: disable=unused-import

from __future__ import absolute_import


from ._resonance import dynamicWidthForBreitWignerS
from ._resonance import breitWigner, breitWignerS, breitWignerNoBarriersupressionFactor, gounariSakurai, breitWignerAlternativeDynwidthS,\
    twoChannelBreitWigner, twoChannelBreitWignerS, multiChannelBreitWigner, multiChannelBreitWignerS, constantWidthBreitWigner, constantWidthBreitWignerS, \
    gounariSakuraiAlternativeDynwidthS, gounariSakuraiS, gounariSakuraiFS, nonrelativisticBreitWignerS, nonrelativisticBreitWinger
from ._swaveAmplitudes import glass, lass, kpisPelaezRodas
from ._swaveAmplitudePalanoPennington import palanoPenningtonKpiKpi, palanoPenningtonKetaKpi
from ._swaveAmplitudeAuMorganPennington import piPiSwaveAuMorganPennington, piPiSwaveAuMorganPenningtonKachaev
from ._swaveAmplitudeFlatte import flatte, flatteS, flatteNorm, flatteNormS
from ._a1_1420_triangle1pAmplitude import a1_1420_triangleAmplitude, a1_1420_triangleMulA1Amplitude
from ._kMatrixFormalism import KMatrixFormalismParams, KMatrixFormalism
