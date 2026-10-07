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
#//      main build file for sphysics
#//
#//
#// Author List:
#//      Sebastian Neubert    TUM            (original author)
#//
#//
#//-------------------------------------------------------------------------
#
# Modifications in sphysics: adapted as main build file for sphysics
# (2022-03-04) and developed further since then, in particular switched to
# pybind11 (2023-05-02), adapted for the installation with pip and reformatted
# (2024-03-22), dependencies on ROOTPWA and other packages removed and download
# of Eigen during the build added (2026-09-23).
# cmake-format: on

# check if cmake has the required version
message(STATUS ">>> This is CMake version ${CMAKE_VERSION}.")
cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
cmake_policy(VERSION 3.12)

# set verbosity
set(CMAKE_VERBOSE_MAKEFILE 0) # if set to 1 compile and link commands are
                              # displayed during build the same effect can be
                              # achieved by calling 'make VERBOSE=1'
option(DEBUG_OUTPUT "en/disable debug output" OFF)

# define project
message(STATUS "")
message(STATUS ">>> Setting up project 'sphysics'.")
project(SPHYSICS)

# set path, where to look first for cmake modules, before ${CMAKE_ROOT}/Modules/
# is checked
message(STATUS "")
message(STATUS ">>> Setting up Cmake modules.")
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmakeModules")
message(STATUS "Using cmake module path '${CMAKE_MODULE_PATH}'.")
# load cmake macros
include(CommonMacros)
include(FeatureSummary)
include(ProcessorCount)

# On Apples the default filename extension of a shared library is ".dylib",
# unfortunately ROOT libraries end with ".so" so Cmake's mechanism of
# determining whether to add a rpath breaks. Also add ".so" to the filetypes to
# be considered as a shared library.
if(APPLE AND "${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".dylib")
  list(APPEND CMAKE_EXTRA_SHARED_LIBRARY_SUFFIXES ".so")
endif()

# force out-of-source builds.
enforce_out_of_source_build()

# Check if ROOT exists
if(DEFINED ENV{ROOTSYS})
  # setup ROOT includes and libraries
  message(STATUS "")
  message(STATUS ">>> Setting up ROOT.")
  find_package(
    ROOT
    6.14.00
    REQUIRED
    "Minuit2"
    "MathMore"
    "TreePlayer"
    "Hist"
    "ROOTTPython"
    "GenVector")
  set_package_properties(
    ROOT PROPERTIES
    DESCRIPTION "C++ HEP analysis framework"
    URL "http://root.cern.ch/"
    PURPOSE "Used for data storage and retrieval"
    TYPE REQUIRED)

  # compiler standard
  set(SUPPORTED_CXX_STANDARDS 11 14 17)
  if(NOT ROOT_CPP_STD IN_LIST SUPPORTED_CXX_STANDARDS)
    message(
      FATAL_ERROR
        "The C++ standard '${ROOT_CPP_STD}' with which ROOT was compiled must be one of '${SUPPORTED_CXX_STANDARDS}'."
    )
  endif()
  set(CMAKE_CXX_STANDARD ${ROOT_CPP_STD})
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(PYTHON_VERSION
      "${ROOT_PYTHON_MAJOR_VERSION}.${ROOT_PYTHON_MINOR_VERSION}")
else()
  set(PYTHON_VERSION "")
  # compiler standard
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# abort, if system is not UNIX
message(STATUS "")
message(STATUS ">>> Setting up system environment.")
if(NOT UNIX)
  message(FATAL_ERROR "!!! This is an unsupported system.")
endif()
ProcessorCount(NMB_CPU_CORES)
message(
  STATUS
    "Detected host system '${CMAKE_HOST_SYSTEM_NAME}' version "
    "'${CMAKE_HOST_SYSTEM_VERSION}', architecture '${CMAKE_HOST_SYSTEM_PROCESSOR}', "
    "${NMB_CPU_CORES} CPU core(s).")
message(
  STATUS
    "Compiling for system '${CMAKE_SYSTEM_NAME}' version '${CMAKE_SYSTEM_VERSION}', "
    "architecture '${CMAKE_SYSTEM_PROCESSOR}'.")

# warn user about outdated or unsupported compiler
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.3)
    message(
      WARNING "??? Your GCC version ${CMAKE_CXX_COMPILER_VERSION} is too low. "
              "GCC version 7.3 or higher is required. Proceed at your own risk."
    )
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.4)
    message(
      WARNING
        "??? Your Clang version ${CMAKE_CXX_COMPILER_VERSION} is too low. "
        "Clang version 3.4 or higher is required. Proceed at your own risk.")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8.0)
    message(
      WARNING
        "??? Your AppleClang version ${CMAKE_CXX_COMPILER_VERSION} is too low. "
        "AppleClang version 8.0 or higher is required. Proceed at your own risk."
    )
  endif()
else()
  message(
    WARNING
      "??? '${CMAKE_CXX_COMPILER_ID} V${CMAKE_CXX_COMPILER_VERSION}' "
      "is an unsupported compiler. Compilation has only been tested with GCC, Clang and AppleClang. "
      "Proceed at your own risk.")
endif()

# define build types predefined build types: "DEBUG" "RELEASE" "RELWITHDEBINFO"
# "MINSIZEREL" set a default build type for single-configuration CMake
# generators, if no build type is set.
message(STATUS "")
message(STATUS ">>> Setting up compiler environment.")
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  message(
    STATUS
      "No build type was specified. Setting build type to 'RELWITHDEBINFO'.")
  set(CMAKE_BUILD_TYPE RELWITHDEBINFO)
endif()

# set common compiler flags consider flags -pedantic -Wsuggest-attribute=pure
# -Wsuggest-attribute=noreturn -Wsuggest-attribute=const
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -Woverloaded-virtual -Werror")
# options switchable via command line enable option by running cmake .. -D
# <VAR>=<any value>
if(ARCH_NATIVE)
  # !Note! may cause trouble in heterogeneous environments `gcc -march=native -Q
  # --help=target` shows the compiler switches set by this option on your
  # machine
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()
# set additional flags for specific build types
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
# consider adding -DNODEBUG (disables all asserts) and -ffast-math
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# report global build settings
foreach(_LANG "C" "CXX")
  message(
    STATUS
      "${_LANG} compiler ID is '${CMAKE_${_LANG}_COMPILER_ID} V${CMAKE_${_LANG}_COMPILER_VERSION}'."
  )
  message(STATUS "Using ${_LANG} compiler '${CMAKE_${_LANG}_COMPILER}'.")
  message(
    STATUS "Using general ${_LANG} compiler flags '${CMAKE_${_LANG}_FLAGS}'.")
endforeach()
unset(_LANG)
message(STATUS "Build type is '${CMAKE_BUILD_TYPE}'.")
message(
  STATUS
    "Using additional CXX compiler flags '${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}' "
    "for build type ${CMAKE_BUILD_TYPE}.")
message(
  STATUS
    "Using additional linker flags '${CMAKE_CXX_LDFLAGS_${CMAKE_BUILD_TYPE}}' "
    "for build type ${CMAKE_BUILD_TYPE}.")
if(DEBUG_OUTPUT)
  include(CMakePrintSystemInformation)
endif()
message(STATUS "Includes defined via environment variables:")
message(STATUS "    CPATH (corresponds to -I...): '$ENV{CPATH}'")
foreach(_LANG "C" "CPLUS")
  message(
    STATUS
      "    ${_LANG}_INCLUDE_PATH (corresponds to -isystem ...): '$ENV{${_LANG}_INCLUDE_PATH}'"
  )
endforeach()
unset(_LANG)

# enable ccache if not already enabled by symlink masquerading and if no other
# CMake compiler launchers are already defined calling 'cmake
# -DDISABLE_CCACHE=ON ..' will disable ccache in any case see
# https://crascit.com/2016/04/09/using-ccache-with-cmake/
# https://stackoverflow.com/a/37828605 https://stackoverflow.com/a/34317588
option(DISABLE_CCACHE "If set to ON, ccache will not be used" OFF)
set(CCACHE_FIND_VERSION "3.1.9")
message(STATUS "")
message(STATUS ">>> Checking for ccache.")
find_program(CCACHE_EXECUTABLE ccache)
if(NOT CCACHE_EXECUTABLE)
  message(STATUS "Cannot find ccache in system paths.")
endif()
if(CCACHE_EXECUTABLE)
  execute_process(
    COMMAND ${CCACHE_EXECUTABLE} --version
    OUTPUT_VARIABLE _CCACHE_VERSION_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  parse_version_from_multline_string("${_CCACHE_VERSION_RAW}"
                                     "ccache[ \t]+version[ \t]+" CCACHE_VERSION)
  message(STATUS "Found ccache version ${CCACHE_VERSION}.")
  unset(_CCACHE_VERSION_RAW)
  if(NOT CCACHE_VERSION)
    message(STATUS "Cannot determine ccache version. Not using ccache.")
  elseif(CCACHE_VERSION VERSION_LESS ${CCACHE_FIND_VERSION})
    message(
      STATUS
        "ccache version ${CCACHE_VERSION} is lower than requested version ${CCACHE_FIND_VERSION}. "
        "Not using ccache.")
  elseif(DISABLE_CCACHE)
    message(STATUS "ccache is disabled via DISABLE_CCACHE.")
  else()
    foreach(_LANG "C" "CXX")
      # expand path to actual compiler executable
      execute_process(
        COMMAND readlink -f "$(which \"${CMAKE_${_LANG}_COMPILER}\")"
        OUTPUT_VARIABLE _COMPILER_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(DEFINED CMAKE_${_LANG}_COMPILER_LAUNCHER)
        message(
          STATUS
            "The launcher '${CMAKE_${_LANG}_COMPILER_LAUNCHER}' is already defined. Leave it unchanged."
        )
      elseif(CMAKE_${_LANG}_COMPILER MATCHES ".*/ccache.*"
             OR _COMPILER_PATH MATCHES ".*/ccache.*")
        # checking for "ccache" anywhere in the compiler path might be too
        # strong more restrictive would be ".*/ccache$", which requires "ccache"
        # at the end
        message(STATUS "ccache seems to be already in use for ${_LANG}.")
      else()
        message(STATUS "Enabling ccache for ${_LANG}.")
        set(CMAKE_${_LANG}_COMPILER_LAUNCHER
            ${CCACHE_EXECUTABLE}
            CACHE STRING "")
      endif()
      unset(_LANG)
      unset(_COMPILER_PATH)
    endforeach()
  endif()
endif()

# redirect output files
message(STATUS "")
message(STATUS ">>> Setting up output paths.")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
message(STATUS "Using library output path '${LIBRARY_OUTPUT_PATH}'.")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
message(STATUS "Using executable output path '${EXECUTABLE_OUTPUT_PATH}'.")

# setup Python
message(STATUS "")
message(STATUS ">>> Setting up Python.")
find_package(Python ${PYTHON_VERSION} REQUIRED COMPONENTS Interpreter
                                                          Development)
set_package_properties(
  Python PROPERTIES
  DESCRIPTION "General-purpose, high-level programming language"
  URL "http://www.python.org"
  PURPOSE "Provides scripting interface to sphysics classes"
  TYPE REQUIRED)
set(Python_LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/pyLib")
message(
  STATUS "Using python bindings output path '${Python_LIBRARY_OUTPUT_PATH}'.")

# setup pybind11
message(STATUS "")
message(STATUS ">>> Setting up pybind11.")
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Using pybind11 version '${pybind11_VERSION}'.")
message(STATUS "Using pybind11 include directory '${pybind11_INCLUDE_DIRS}'.")

# setup tensorflow
message(STATUS "")
message(STATUS ">>> Setting up tensorflow.")
find_package(Tensorflow 2.0 REQUIRED)
set_package_properties(
  Tensorflow PROPERTIES
  DESCRIPTION "Machine learning platform"
  URL "https://www.tensorflow.org/"
  PURPOSE "Used for generating the compute graph."
  TYPE REQUIRED)

# setup pylint (optional)
message(STATUS "")
message(STATUS ">>> Setting up pylint.")
find_package(PYLINT 1.7.1)
set_package_properties(
  PYLINT PROPERTIES
  DESCRIPTION "code analysis for Python"
  URL "https://www.pylint.org"
  PURPOSE "Used to assess quality of Python code."
  TYPE OPTIONAL)
if(NOT PYLINT_FOUND)
  message(
    STATUS
      "Cannot find pylint executable. Quality of python code will no be assessed."
  )
endif()

# make git hash accessible via predefined macro GIT_HASH
message(STATUS "")
message(STATUS ">>> Setting up Git.")
find_package(Git)
set_package_properties(
  Git PROPERTIES
  DESCRIPTION "Free and open source distributed version control system"
  URL "http://git-scm.com/"
  PURPOSE "Used to determine repository hash that is put into executables"
  TYPE RECOMMENDED)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} log --pretty="%H" -n1 -- ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_HASH
    RESULT_VARIABLE _GIT_LOG_RETURN
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _GIT_LOG_RETURN)
    message(STATUS "Git repository hash is '${GIT_HASH}'.")
  else()
    message(
      Warning "??? Running 'git' did not succeed. Repository hash unknown.")
  endif()
endif()

message(STATUS "")
message(STATUS ">>> Setting up modernplotting.")
pybind11_find_import(modernplotting VERSION 0.1.0)
set_package_properties(
  modernplotting PROPERTIES
  DESCRIPTION "Plotting library"
  URL ""
  PURPOSE "Used for plotting"
  TYPE RECOMMENDED)

enable_testing()

# define include and library paths for project-internal libraries
#
# the following scheme is used:
#
# * for each source folder a shared library is generated
#
# * the name 'libSphysics<Folder>.so' of the library is derived from the
#   directory name, e.g. the code in the 'foo' directory will be compiled into
#   'libSphysicsFoo.so'.
#
# * for each source folder two CMake variables are defined:
#   SPHYSICS_<FOLDER>_INCLUDE_DIR that points to the source directory and
#   SPHYSICS_<FOLDER>_LIB that holds the library name
#
# * compilation of each directory can be switched off by defining a variable
#   with name DO_NOT_COMPILE_SPHYSICS_<FOLDER> and setting it to a non-empty
#   value
#
# * each folder may contain one linkdef.h file, which will be used to generate
#   ROOT dictionaries (dict.cc)

# dependencies

# setup Eigen, which is downloaded during the configuration
message(STATUS "")
message(STATUS ">>> Setting up Eigen.")
if(CMAKE_VERSION VERSION_LESS 3.18)
  message(FATAL_ERROR "CMake 3.18 or higher is required to download Eigen.")
endif()
include(FetchContent)
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  URL_HASH
    SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  # only download Eigen, but do not add its CMake project
  SOURCE_SUBDIR do-not-add-eigen-cmake-project)
FetchContent_MakeAvailable(eigen)
set(DEPS_CPPLIB_INCLUDE_DIR "${eigen_SOURCE_DIR}")
message(STATUS "Using Eigen 3.4.0 from '${DEPS_CPPLIB_INCLUDE_DIR}'.")

message(STATUS "")
message(STATUS ">>> Defining include and library paths.")
set(_SRC_DIRS
    "src/sphysics" "src/cppLib/utilities" "src/cppLib/pwa"
    "src/cppLib/constants" "src/cppLib/particleData" "src/cppLib/pyBindings")
# loop over all source directories
set(SPHYSICS_CPPLIB_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/cppLib")
foreach(_SRC_DIR ${_SRC_DIRS})
  var_name_from_dir_name(${_SRC_DIR} _VAR_NAME)
  if(NOT DO_NOT_COMPILE_${_VAR_NAME})
    lib_name_from_dir_name(${_SRC_DIR} _LIB_NAME)
    set(${_VAR_NAME}_SRC_DIR "${CMAKE_SOURCE_DIR}/${_SRC_DIR}")
    set(${_VAR_NAME}_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/${_SRC_DIR}")
    set(${_VAR_NAME}_LIB "${_LIB_NAME}")
    message(
      STATUS
        ">>> Directory '${_SRC_DIR}' will be compiled into library '${_LIB_NAME}', "
        "setting up variables '${_VAR_NAME}_SRC_DIR', '${_VAR_NAME}_INCLUDE_DIR' and, '${_VAR_NAME}_LIB'."
    )
    unset(_LIB_NAME)
  endif()
  unset(_VAR_NAME)
endforeach()

# build project-internal libraries
foreach(_SRC_DIR ${_SRC_DIRS})
  var_name_from_dir_name(${_SRC_DIR} _VAR_NAME)
  if(NOT DO_NOT_COMPILE_${_VAR_NAME})
    add_subdirectory("${_SRC_DIR}")
  endif()
  unset(_VAR_NAME)
endforeach()
unset(_SRC_DIRS)

add_subdirectory("scripts")

add_subdirectory("test")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/data/
        DESTINATION ${SKBUILD_PROJECT_NAME}/data/)

# print package and feature summary
message(STATUS "")
message(STATUS ">>> Package summary:")
feature_summary(
  WHAT PACKAGES_FOUND
  FATAL_ON_MISSING_REQUIRED_PACKAGES
  DESCRIPTION "Found packages:")
feature_summary(
  WHAT PACKAGES_NOT_FOUND
  FATAL_ON_MISSING_REQUIRED_PACKAGES
  DESCRIPTION "Packages not found:")

message(STATUS "")
message(STATUS ">>> Done.")
message(STATUS "Run 'make' to build the project.")
if(${NMB_CPU_CORES} GREATER 1)
  message(
    STATUS
      "On this machine using 'make -j${NMB_CPU_CORES}' can speedup compilation considerably."
  )
endif()
