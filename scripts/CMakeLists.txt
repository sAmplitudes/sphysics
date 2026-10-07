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
#//      build file for the sphysics scripts
#//
#//
#// Author List:
#//      Boris Grube          TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: handling of the Python scripts from ROOTPWA's
# pyInterface/CMakeLists.txt adapted for the sphysics scripts (2022-03-04) and
# developed further since then, adapted for the installation with pip and
# reformatted (2024-03-22).
# cmake-format: on

message_setup_this_dir()

# scripts
set(SPHYSICS_PYTHON_SCRIPTS_FILES convertVariablesHdf5ToParquet)

set(SPHYSICS_PYTHON_SCRIPTS_EXEC)
foreach(_FILE ${SPHYSICS_PYTHON_SCRIPTS_FILES})
  get_filename_component(_FILENAME "${_FILE}" NAME_WE)
  set(_FILENAME "sphysics.${_FILENAME}")
  # list of files to link
  list(APPEND SPHYSICS_PYTHON_SCRIPTS_EXEC
       "${EXECUTABLE_OUTPUT_PATH}/${_FILENAME}")
  add_custom_command(
    OUTPUT ${EXECUTABLE_OUTPUT_PATH}/${_FILENAME}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE}
            ${EXECUTABLE_OUTPUT_PATH}/${_FILENAME}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE})
endforeach()
unset(_FILE)
unset(_FILENAME)

# target to link the python files
add_custom_target(sphysicsScripts ALL DEPENDS ${SPHYSICS_PYTHON_SCRIPTS_EXEC})

# pylint tests
if(PYLINT_FOUND)

  # check scripts
  foreach(_FILE ${SPHYSICS_PYTHON_SCRIPTS_FILES})
    get_filename_component(_FILENAME ${_FILE} NAME_WE)
    add_test(
      NAME pyLint${_FILENAME}
      COMMAND
        ${PYLINT_EXECUTABLE} --reports=n
        --rcfile=${SPHYSICS_SPHYSICS_SRC_DIR}/pylintrc
        ${CMAKE_CURRENT_SOURCE_DIR}/${_FILE})
    set_property(
      TEST pyLint${_FILENAME}
      APPEND
      PROPERTY ENVIRONMENT
               "PYTHONPATH=${Python_LIBRARY_OUTPUT_PATH}:$ENV{PYTHONPATH}")
  endforeach()
  unset(_FILE)
  unset(_FILENAME)
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/thisSphysics.sh.in
               ${EXECUTABLE_OUTPUT_PATH}/thisSphysics.sh @ONLY)
add_custom_target(thisSphysics ALL
                  DEPENDS ${EXECUTABLE_OUTPUT_PATH}/thisSphysics.sh)
