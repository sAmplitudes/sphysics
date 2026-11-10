sphysics
========

sphysics is a Python package with a C++ core for amplitude analyses in particle physics.
It is developed for amplitude analyses at the Belle II experiment.
sphysics provides tools for amplitude analysis, event selection, kinematic calculations, machine learning, and more:

- :doc:`Amplitudes <sphysics.amplitudes>`: dynamic amplitudes of resonances, e.g., Breit-Wigner, Gounaris-Sakurai, Flatté, LASS, and K-matrix parameterizations.
- :doc:`Partial-wave analysis <sphysics.pwa>`: centrifugal-barrier factors, models for two-body and three-body decays and for Dalitz plots, polarimeters for tau pairs, and the fitting of such models with TensorFlow and iminuit.
- :doc:`Kinematics <sphysics.kinematics>`, :doc:`Lorentz vectors <sphysics.lorentz>`, and :doc:`Dirac spinors <sphysics.dirac>`: break-up momenta, phase-space factors, rotations, boosts, and spinors.
- :doc:`Event selection <sphysics.eventselection>`: loading variables from ROOT trees, histograms, efficiency tables, threshold scans, and plotting.
- :doc:`Generators <sphysics.generators>`: phase-space generators for two-body and n-body decays, including tau decays.
- :doc:`Machine learning <sphysics.torch>`: training utilities, Lightning modules, metrics, and neural networks for hadron identification, based on PyTorch.
- :doc:`Math <sphysics.math>`, :doc:`random numbers <sphysics.random>`, and :doc:`utilities <sphysics.utils>`: functions that accept NumPy as well as TensorFlow arguments, seeding, logging, and timing.
- :doc:`Constants <constants>`: physics constants and particle properties read from a particle data table.

Performance-critical parts are implemented in C++ and made available in Python via pybind11.


Documentation
-------------

- :doc:`API reference <sphysics>`: the modules, classes, and functions of sphysics.
- :ref:`Using sphysics <getting-started>`: how to install and use sphysics.
- :ref:`Conventions <conventions>`: conventions for vectors, Lorentz vectors, and matrices.
- :ref:`Contributing <contributing>`: how to contribute to sphysics, test changes, and release new versions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   API reference <sphysics>
   gettingstarted
   conventions
   contributing


Citing sphysics
---------------

If you use sphysics, please cite it using its DOI `10.5281/zenodo.23214121 <https://doi.org/10.5281/zenodo.23214121>`_, which refers to all versions; the DOIs of the individual versions are listed on Zenodo.
The citation metadata are also available in `CITATION.cff <https://github.com/sAmplitudes/sphysics/blob/main/CITATION.cff>`_.


License
-------

sphysics is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
sphysics is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the `LICENSE <https://github.com/sAmplitudes/sphysics/blob/main/LICENSE>`_ file for the full license text.

Copyright (C) 2022-2026 the sphysics authors, see `AUTHORS.md <https://github.com/sAmplitudes/sphysics/blob/main/AUTHORS.md>`_.


Third-party software
--------------------

sphysics uses `Eigen <https://libeigen.gitlab.io>`_ 3.4.0, which is downloaded when building sphysics.
Compiled sphysics packages contain code of Eigen, which is licensed under the Mozilla Public License 2.0, with some parts licensed under the BSD and Apache-2.0 licenses.
The source code of Eigen 3.4.0 is available at https://gitlab.com/libeigen/eigen/-/tree/3.4.0.
Compiled sphysics packages also contain code of `pybind11 <https://github.com/pybind/pybind11>`_, which is licensed under the BSD 3-Clause License.
