# Authors

sphysics is developed by Stefan Wallner (main author), with contributions by
Martin Bartl, Julien Beckers, Yannik Fausch, Godo Kurten, Fabian Krinner,
Claudia Perez-Orive, Markus Reif, Xavier Simo, Oskar Tittel, and Miriam
Weiskopf.

## Code from other authors and projects

- [ROOTPWA](https://github.com/ROOTPWA-Maintainers/ROOTPWA) (GNU GPL version 3
  or later): the build system (`CMakeLists.txt` files and `cmakeModules/`), the
  particleData library and the particle data tables (`src/cppLib/particleData/`,
  `data/particleDatatable/`), and the Python bindings of the particle data
  (`particleProperties_py.cc` and `particleDataTable_py.cc` in
  `src/cppLib/pyBindings/constants/`), which are based on ROOTPWA's boost.python
  bindings. Their authors are named in the file headers.
- [basf2](https://github.com/belle2/basf2) (GNU LGPL version 3 or later): the
  particle names and PDG codes in `src/sphysics/_constants/_pdgcodes.py`.

## Data

- The particle properties in the particle data tables are taken from the Review
  of Particle Physics of the Particle Data Group.
- `data/amplitudes/K0K0barspinnolam.csv` and `KplusKminusspinnolam.csv`:
  triangle amplitude of the a1(1420), provided by the COMPASS Collaboration and
  published with permission, see Phys. Rev. Lett. 127, 082501 (2021).
- `data/amplitudes/kpiS_Pelaez-Rodas_2010.11222.csv`: Kπ S-wave amplitude of J.
  R. Peláez and A. Rodas, Phys. Rept. 969 (2022) 1 (arXiv:2010.11222).
