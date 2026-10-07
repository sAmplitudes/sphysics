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
#//      build file for tfLib
#//
#//
#// Author List:
#//      Boris Grube          TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: written for tfLib based on ROOTPWA's library build
# files (2022-03-04), installation with pip added (2024-03-22), dependencies on
# ROOTPWA and ROOT removed (2026-09-23).
# cmake-format: on

message_setup_this_dir()

# set include directories
include_directories(
	${SPHYSICS_FIELD_INCLUDE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}
	SYSTEM
	${Tensorflow_INCLUDE_DIR}
	)



# source files that are compiled into library
file(GLOB SOURCES
	"*.cpp"
	"*.c"
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Tensorflow_CXX_FLAGS}")

# library
lib_name_for_current_dir(THIS_LIB)

link_directories(${Tensorflow_LIBRARY_DIR})

make_shared_library(
	"${THIS_LIB}"
	"${SOURCES}"
	"${Tensorflow_LIBS}"
	GSL::gsl
	)

unset(_Tensorflow_COMPILE)
install(TARGETS ${THIS_LIB} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
