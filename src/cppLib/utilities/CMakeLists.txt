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
#//      build file for utilities
#//
#//
#// Author List:
#//      Boris Grube          TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: adapted for sphysics (2022-03-04), boost dependency
# removed (2023-05-03), installation with pip added (2024-03-22), ROOT
# dependency removed (2026-09-23).
# cmake-format: on


message_setup_this_dir()


# set include directories
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	SYSTEM
	)



# source files that are compiled into library
file(GLOB SOURCES
	"*.cpp"
	"*.c"
)



# make environment variables accessible from program code
ADD_CUSTOM_TARGET(
	environment
	${CMAKE_COMMAND} -D SRC=${CMAKE_CURRENT_SOURCE_DIR}/environmentStatus.h.in
	                 -D DST=${CMAKE_CURRENT_BINARY_DIR}/environmentStatus.h
	                 -D CMAKE_HOST_SYSTEM_NAME=${CMAKE_HOST_SYSTEM_NAME}
	                 -D CMAKE_HOST_SYSTEM_PROCESSOR=${CMAKE_HOST_SYSTEM_PROCESSOR}
	                 -D CMAKE_HOST_SYSTEM_VERSION=${CMAKE_HOST_SYSTEM_VERSION}
	                 -D NMB_CPU_CORES=${NMB_CPU_CORES}
	                 -D HOSTNAME=${HOSTNAME}
	                 -D USER=${USER}
	                 -D SPHYSICS_CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}
	                 -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
	                 -D Boost_VERSION=${Boost_VERSION}
	                 -D Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}
	                 -D YamlCpp_VERSION=${YamlCpp_VERSION}
	                 -D YamlCpp_LIBS=${YamlCpp_LIBS}
	                 -D Tensorflow_VERSION=${Tensorflow_VERSION}
	                 -D PYTHONLIBS_VERSION=${PYTHONLIBS_VERSION}
	                 -D PYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS}
	                 -P ${CMAKE_CURRENT_SOURCE_DIR}/environment.cmake
)

# library
lib_name_for_current_dir(THIS_LIB)
make_shared_library(
	"${THIS_LIB}"
	"${SOURCES}"
	)
add_dependencies(${THIS_LIB} environment)
install(TARGETS ${THIS_LIB} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
