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
#//      build file for constants
#//
#//
#// Author List:
#//      Boris Grube          TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: written for constants based on ROOTPWA's library
# build files (2022-10-31), boost dependency removed (2023-05-03), installation
# with pip added (2024-03-22), ROOT dependency removed (2026-09-23).
# cmake-format: on


message_setup_this_dir()


# set include directories
include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_BINARY_DIR}
	${SPHYSICS_CPPLIB_INCLUDE_DIR}
	)



# source files that are compiled into library
file(GLOB SOURCES
	"*.cpp"
	"*.c"
)

# library
lib_name_for_current_dir(THIS_LIB)
make_shared_library(
	"${THIS_LIB}"
	"${SOURCES}"
	)
add_dependencies(${THIS_LIB} environment)
install(TARGETS ${THIS_LIB} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
