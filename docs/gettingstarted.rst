.. _getting-started:

Using sphysics
==============

This module can be installed using pip (See Getting Started).
Within your Python environment, where you installed sphysics, you can use it with:

.. code-block:: python

   import sphysics


Getting Started
-------------------------------------

In order to use sphysics, please follow these steps:

1. Clone the repository.
2. *(Optional)* Set the SPHYSICS_PARTICLEDATATABLE to the path where your particle data table is stored.
   You can obtain the particle data table file from the data folder, for example:

   .. code-block:: bash

      export SPHYSICS_PARTICLEDATATABLE="$SPHYSICS/data/particleDatatable/particleDataTable2022.txt"

   This step is optional as sphysics automatically uses particleDataTable2022.txt .
3. Set up your development environment with the Python version that you want to use.
4. You can install sphysics by executing:

   .. code-block:: bash

      pip install .

   in the base directory of sphysics.


Alternative Installation from the GitLab or GitHub Repository
--------------------------------------------------------------

Alternatively, sphysics can be directly installed from the GitLab repository via pip using:

.. code-block:: bash

   pip install git+ssh://git@gitlab.desy.de/belle2/physics/amplitudesatbelleii/sphysics.git

Without access to the GitLab repository, sphysics can be installed from its public GitHub repository (https://github.com/sAmplitudes/sphysics) using:

.. code-block:: bash

   pip install git+https://github.com/sAmplitudes/sphysics.git


Comments
--------------

- sphysics is distributed as a source package, i.e. its C++ code is compiled during the installation.
  This requires a Linux system with a C++ compiler supporting C++17.
  Unless the pip build isolation is disabled (see below), pip downloads the build dependencies listed in ``pyproject.toml``, including TensorFlow, during the installation.

- Building sphysics downloads Eigen 3.4.0 (https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz), which requires network access.
  To build sphysics without network access, download and extract the archive beforehand and pass its path to the build:

  .. code-block:: bash

     pip install --config-settings=cmake.define.FETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/eigen-3.4.0 .

- If you are using a non-system software environment, such as Python and compilers from basf2 (cvmfs) instead of the system compiler,
  you have to disable the pip build isolation. To do this, first install all build dependencies:

  .. code-block:: bash

     pip install pybind11 scikit-build-core 'cmake>=3.18' pyproject_metadata setuptools_scm tensorflow pathspec

  Then, install sphysics via:

  .. code-block:: bash

     pip install --no-build-isolation .

- To use the torch module, please install torch, torchmetrics, torchinfo, and lightning first.
  The torch module is not imported automatically, so you have to import it manually:

  .. code-block:: python

     import sphysics.torch

- The plotting functions require modernplotting, which is not publicly available.
  With access to its GitLab repository, it can be installed from there (https://gitlab.desy.de/belle2/physics/amplitudesatbelleii/softwarestack/modernplotting).
  If modernplotting is not installed, the functions ``plotVariablesDistributions``, ``plotVariablesDistributions2D``, ``compareVariablesDistributions``, and ``thresholdScanPlots`` of ``sphysics.eventselection``,
  ``sphysics.torch.plotSequentialModel``, and ``sphysics.torch.lightning.plotMetric`` are not available, and ``sphysics.torch.hadronID.showModel`` raises a ``NotImplementedError``.
  The plotting functions of ``sphysics.torch.hadronID`` require a modernplotting style object.
