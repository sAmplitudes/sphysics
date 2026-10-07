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
#//      build file for the sphysics Python package
#//
#//
#// Author List:
#//      Boris Grube          TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: handling of the Python package from ROOTPWA's
# pyInterface/CMakeLists.txt adapted for the sphysics Python package
# (2022-03-04) and developed further since then, rewritten for the installation
# with pip and reformatted (2024-03-22).
# cmake-format: on

message_setup_this_dir()

# 'sphysics' module
file(
  GLOB_RECURSE SPHYSICS_PYTHON_PACKAGE_FILES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.py)

# remove plotting if modernplotting is not available
if(NOT modernplotting_FOUND)
  list(FILTER SPHYSICS_PYTHON_PACKAGE_FILES EXCLUDE REGEX ".*plotting.*")
endif()

set(SPHYSICS_PYTHON_PACKAGE_BYTECODES)
set(SPHYSICS_PYTHON_PACKAGE_DIRS)
set(SPHYSICS_PYTHON_PACKAGE_LINKS)
foreach(_FILE ${SPHYSICS_PYTHON_PACKAGE_FILES})
  get_filename_component(_DIR "${_FILE}" DIRECTORY)
  get_filename_component(_NAMEWLE "${_FILE}" NAME_WLE)
  # list of directories to create
  list(APPEND SPHYSICS_PYTHON_PACKAGE_DIRS
       "${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_DIR}")
  add_custom_command(
    OUTPUT ${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_DIR}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_DIR})
  # list of files to copy
  list(APPEND SPHYSICS_PYTHON_PACKAGE_LINKS
       "${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_FILE}")
  add_custom_command(
    OUTPUT ${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE}
            ${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_FILE}
    DEPENDS pySphysicsPackageDirs ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE})
  # list of files containing bytecode
  list(
    APPEND
    SPHYSICS_PYTHON_PACKAGE_BYTECODES
    "${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_DIR}/__pycache__/${_NAMEWLE}.cpython-${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.pyc"
  )
  add_custom_command(
    OUTPUT
      ${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_DIR}/__pycache__/${_NAMEWLE}.cpython-${Python_VERSION_MAJOR}${Python_VERSION_MINOR}.pyc
    COMMAND ${Python_EXECUTABLE} -m py_compile
            ${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_FILE}
    DEPENDS pySphysicsPackageLinks
            ${Python_LIBRARY_OUTPUT_PATH}/sphysics/${_FILE})
endforeach()
unset(_DIR)
unset(_FILE)

# target to create the module directory structure
add_custom_target(pySphysicsPackageDirs ALL
                  DEPENDS ${SPHYSICS_PYTHON_PACKAGE_DIRS})
# target to link the python files
add_custom_target(
  pySphysicsPackageLinks ALL DEPENDS pySphysicsPackageDirs
                                     ${SPHYSICS_PYTHON_PACKAGE_LINKS})
# target to create python bytecode
add_custom_target(
  pySphysicsPackageBytecodes ALL DEPENDS pySphysicsPackageLinks
                                         ${SPHYSICS_PYTHON_PACKAGE_BYTECODES})

# copy the shared library with the python bindings
add_custom_target(
  libSphysicsPyLink ALL
  COMMAND
    ${CMAKE_COMMAND} -E copy
    ${LIBRARY_OUTPUT_PATH}/_core.cpython-${Python_VERSION_MAJOR}${Python_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu${CMAKE_SHARED_LIBRARY_SUFFIX}
    ${Python_LIBRARY_OUTPUT_PATH}/sphysics/_core.cpython-${Python_VERSION_MAJOR}${Python_VERSION_MINOR}-${CMAKE_SYSTEM_PROCESSOR}-linux-gnu${CMAKE_SHARED_LIBRARY_SUFFIX}
  DEPENDS _core pySphysicsPackageDirs)

# copy the shared library with the tensorflow C++ implementations
# add_custom_target( libSphysicsTfLibLink ALL COMMAND ${CMAKE_COMMAND} -E copy
# ${LIBRARY_OUTPUT_PATH}/${CMAKE_SHARED_LIBRARY_PREFIX}SphysicsTfLib${CMAKE_SHARED_LIBRARY_SUFFIX}
# ${Python_LIBRARY_OUTPUT_PATH}/sphysics/libSphysicsTfLib.so DEPENDS
# SphysicsTfLib pySphysicsPackageDirs)

# pylint tests
if(PYLINT_FOUND)
  # check 'sphysics' module
  add_test(NAME pyLintPySphysics
           COMMAND ${PYLINT_EXECUTABLE} --reports=n
                   --rcfile=${CMAKE_CURRENT_SOURCE_DIR}/pylintrc sphysics)
  set_property(
    TEST pyLintPySphysics
    APPEND
    PROPERTY ENVIRONMENT
             "PYTHONPATH=${Python_LIBRARY_OUTPUT_PATH}:$ENV{PYTHONPATH}")

endif()
