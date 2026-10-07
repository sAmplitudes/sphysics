# cmake-format: off
#///////////////////////////////////////////////////////////////////////////
#//
#//    Copyright 2010
#//
#//    This file is part of rootpwa
#//
#//    rootpwa is free software: you can redistribute it and/or modify
#//    it under the terms of the GNU General Public License as published by
#//    the Free Software Foundation, either version 3 of the License, or
#//    (at your option) any later version.
#//
#//    rootpwa is distributed in the hope that it will be useful,
#//    but WITHOUT ANY WARRANTY; without even the implied warranty of
#//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#//    GNU General Public License for more details.
#//
#//    You should have received a copy of the GNU General Public License
#//    along with rootpwa.  If not, see <http://www.gnu.org/licenses/>.
#//
#///////////////////////////////////////////////////////////////////////////
#//-------------------------------------------------------------------------
#//
#// Description:
#//      build file for the sphysics Python bindings
#//
#//
#// Author List:
#//      Boris Grube          TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: adapted for the sphysics Python bindings
# (2022-03-04) and developed further since then, switched from boost.python to
# pybind11 (2023-05-02), dependencies on ROOTPWA and ROOT removed (2026-09-23).
# cmake-format: on


message_setup_this_dir()

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${SPHYSICS_CPPLIB_INCLUDE_DIR}
	${DEPS_CPPLIB_INCLUDE_DIR}
	)

set(SOURCES
	sphysicsPy.cc
	constants/constants_py.cc
	constants/particleProperties_py.cc
	constants/particleDataTable_py.cc
	pwa/fitresult_py.cc
	utilities/utilities_py.cc
)

set(THIS_LIB "_core")

# pybind11_add_module(_sphysics_constants constants/constants_py.cc)
# pybind11_add_module(_sphysics_pwa pwa/fitresult_py.cc)
# pybind11_add_module(_sphysics_utilities utilities/utilities_py.cc)

# target_link_libraries(_sphysics_utilities PRIVATE ${SPHYSICS_UTILITIES_LIB} )


pybind11_add_module(${THIS_LIB} ${SOURCES})

target_link_libraries(${THIS_LIB} PRIVATE
	${SPHYSICS_PWA_LIB}
	${SPHYSICS_UTILITIES_LIB}
	${SPHYSICS_PARTICLEDATA_LIB}
	)
install(TARGETS ${THIS_LIB} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})

# set(PYUTILS_SUBDIR pyUtils)
# set(UTILITIES_SUBDIR utilities)
# set(PWA_SUBDIR pwa)
# set(CONSTANTS_SUBDIR constants)


# # set include directories
# include_directories(
# 	${UTILITIES_SUBDIR}
# 	${PWA_SUBDIR}
# 	${CONSTANTS_SUBDIR}
# 	${SPHYSICS_UTILITIES_INCLUDE_DIR}
# 	${SPHYSICS_CPPLIB_INCLUDE_DIR}
# 	SYSTEM
# 	${Boost_INCLUDE_DIRS}
# 	${Libconfig_INCLUDE_DIR}
# 	${Python_INCLUDE_DIRS}
# 	)



# # source files that are compiled into library
# set(SOURCES
# 	sphysicsPy.cc
# 	${UTILITIES_SUBDIR}/environment_py.cc
# 	${UTILITIES_SUBDIR}/numpy_py.cc
# 	${PWA_SUBDIR}/fitresult_py.cc
# 	${CONSTANTS_SUBDIR}/constants_py.cc
# 	)


# # library
# make_shared_library(
# 	"${THIS_LIB}"
# 	"${SOURCES}"
# 	"${Python_LIBRARIES}"
# 	"${Boost_PYTHON${Python_VERSION_MAJOR}${Python_VERSION_MINOR}_LIBRARY}"
# 	"${Boost_NUMPY${Python_VERSION_MAJOR}${Python_VERSION_MINOR}_LIBRARY}"
# 	"${Libconfig_LIBS}"
# 	${SPHYSICS_PHASESPACE_LIB}
# 	${SPHYSICS_UTILITIES_LIB}
# 	${SPHYSICS_PWA_LIB}
# 	)

