.. _contributing:

Contributing and Developing
============

The development of sphysics is carried out at the Belle2 GitLab repository (https://gitlab.desy.de/belle2/physics/amplitudesatbelleii/sphysics).

To include your changes, please follow these steps:

1. Create a new branch
2. Make your changes
3. Test your changes (see below)
4. Commit your changes with a descriptive message to the new branch
5. Push the new branch to the GitLab repository
6. Create a merge request, adding **Stefan Wallner** as a reviewer

If you find any issues or have suggestions for sphysics, please create a GitLab issue.
If you do not have access to the GitLab repository, please report issues on GitHub (https://github.com/sAmplitudes/sphysics/issues).


Test your changes
-------------------------------------
Before committing, run the following command in the base folder of this repository:

.. code-block:: bash

   make test

This will:

1. Make minor changes to meet the code criteria.
2. Check the code.

Please fix all displayed issues before creating a merge request.
pre-commit, which is required for this, is part of the ``dev`` extra, i.e. it can be installed via ``pip install .[dev]``.



Install development mode
-------------------------------------
To install it in development mode (editable install), execute:

.. code-block:: bash

   pip install --editable .

In this way, the source files are not copied to the virtual environment but linked.
Hence, changes in the source files directly reflect in the virtual environment.
Changes in the C++ files will be reflected **only after recompiling the C++ code via pip install**.
