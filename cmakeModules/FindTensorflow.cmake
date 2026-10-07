# cmake-format: off
#///////////////////////////////////////////////////////////////////////////
#//
#//    Copyright 2016
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
#//      cmake module for finding NumPy libraries and include files
#//
#//      following variables are defined:
#//      Tensorflow_FOUND       - indicates whether Tensorflow was found
#//      Tensorflow_VERSION     - version of Tensorflow
#//      Tensorflow_INCLUDE_DIR - include directory for Tensorflow
#//      Tensorflow_LIBRARY_DIR - Tensorflow library directory
#//      Tensorflow_LIBS        - Tensorflow library files
#//      Tensorflow_CXX_FLAGS   - Tensorflow compiler flags
#//
#//      Example usage:
#//          find_package(Tensorflow 1.8 Optional)
#//
#//
#// Author List:
#//      Stefan Wallner       TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: written to find TensorFlow based on ROOTPWA's
# FindNumPy.cmake by Sebastian Uhl (2022-03-04), reformatted (2024-03-22).
# cmake-format: on

set(Tensorflow_FOUND TRUE)
set(Tensorflow_ERROR_REASON "")

set(Tensorflow_VERSION NOTFOUND)
set(Tensorflow_INCLUDE_DIR NOTFOUND)
set(Tensorflow_LIBRARY_DIR NOTFOUND)
set(Tensorflow_LIBS NOTFOUND)
set(Tensorflow_CXX_FLAGS NOTFOUND)

# check for Python
if(NOT Python_FOUND)
  set(Tensorflow_FOUND FALSE)
  set(Tensorflow_ERROR_REASON
      "${Tensorflow_ERROR_REASON} Did not find Python. Python needs to be set up prior to Tensorflow."
  )
endif()
if(NOT Python_EXECUTABLE)
  set(Tensorflow_FOUND FALSE)
  set(Tensorflow_ERROR_REASON
      "${Tensorflow_ERROR_REASON} Did not find executable of Python interpreter. Python needs to be set up prior to Tensorflow."
  )
endif()

if(Python_EXECUTABLE)
  # get version
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c
            "import tensorflow as tf; print(tf.__version__)"
    RESULT_VARIABLE _Tensorflow_IMPORT_SUCCESS
    OUTPUT_VARIABLE _Tensorflow_VERSION_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(_Tensorflow_IMPORT_SUCCESS EQUAL 0)
    # version extracted successfully
    set(Tensorflow_VERSION "${_Tensorflow_VERSION_RAW}")

    # get include path
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c
              "import tensorflow as tf; print(tf.sysconfig.get_include())"
      OUTPUT_VARIABLE _Tensorflow_INCLUDE_DIR_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(Tensorflow_INCLUDE_DIR "${_Tensorflow_INCLUDE_DIR_RAW}")
    unset(_Tensorflow_INCLUDE_DIR_RAW)
    unset(_Tensorflow_INCLUDE_FILE)

    # get library path
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c
              "import tensorflow as tf; print(tf.sysconfig.get_lib())"
      OUTPUT_VARIABLE _Tensorflow_LIBRARY_DIR_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(Tensorflow_LIBRARY_DIR "${_Tensorflow_LIBRARY_DIR_RAW}")
    unset(_Tensorflow_LIBRARY_DIR_RAW)

    # get libs
    execute_process(
      COMMAND
        ${Python_EXECUTABLE} -c
        "import tensorflow as tf; print(' '.join([l.lstrip('-l') for l in tf.sysconfig.get_link_flags()[1:]]))"
      OUTPUT_VARIABLE _Tensorflow_LIBS_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(Tensorflow_LIBS "${_Tensorflow_LIBS_RAW}")
    unset(_Tensorflow_LIBS_RAW)

    # get compiler flags
    execute_process(
      COMMAND
        ${Python_EXECUTABLE} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()[1:]))"
      OUTPUT_VARIABLE _Tensorflow_CXX_FLAGS_RAW
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    set(Tensorflow_CXX_FLAGS "${_Tensorflow_CXX_FLAGS_RAW}")
    unset(_Tensorflow_CXX_FLAGS_RAW)
  endif()
  unset(_Tensorflow_IMPORT_SUCCESS)
  unset(_Tensorflow_VERSION_RAW)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Tensorflow
  FOUND_VAR Tensorflow_FOUND
  REQUIRED_VARS Tensorflow_INCLUDE_DIR Tensorflow_LIBRARY_DIR Tensorflow_LIBS
                Tensorflow_CXX_FLAGS Tensorflow_VERSION
  VERSION_VAR Tensorflow_VERSION)
# additional reporting
if(NOT Tensorflow_FOUND)
  message(
    STATUS
      "Unable to find requested Tensorflow installation:${Tensorflow_ERROR_REASON}"
  )
endif()

# hide variables from normal GUI
mark_as_advanced(Tensorflow_VERSION Tensorflow_INCLUDE_DIR
                 Tensorflow_LIBRARY_DIR Tensorflow_LIBS Tensorflow_CXX_FLAGS)

if(NOT Tensorflow_FOUND)
  unset(Tensorflow_VERSION)
  unset(Tensorflow_INCLUDE_DIR)
  unset(Tensorflow_LIBRARY_DIR)
  unset(Tensorflow_LIBS)
  unset(Tensorflow_CXX_FLAGS)
endif()
