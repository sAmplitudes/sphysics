.. _conventions:

Conventions in sphysics
========================

Here we define some conventions in sphysics.


Vectors, Lorentz Vectors, and Matrices
---------------------------------------

- Vectors are represented as arrays with shape ``(n, ...)``. The first dimension corresponds to the vector components, and the remaining dimensions correspond to the batch dimensions.
- Matrices are represented as arrays with shape ``(m, n, ...)``. The first two dimensions correspond to the matrix dimensions, and the remaining dimensions correspond to the batch dimensions.
- Operations on vectors and matrices can be found in :doc:`sphysics Math <sphysics.math>`.


Lorentz Vectors
~~~~~~~~~~~~~~~~
- Lorentz vectors are represented as arrays with shape ``(4, ...)``. The first dimension corresponds to the components of the Lorentz vector, and the remaining dimensions correspond to the batch dimensions.
- The components of the Lorentz vector are ordered as :math:`(E, p_x, p_y, p_z)`, or  :math:`(t, x, y, z)`, i.e., the time component is first, followed by the spatial components.
- The metric convention is :math:`\eta = \mathrm{diag}(1, -1, -1, -1)`, i.e., the time component has a positive sign and the spatial components have negative signs.
- Operations on lorentz tensors can be found in :doc:`sphysics Lorentz <sphysics.lorentz>`.
