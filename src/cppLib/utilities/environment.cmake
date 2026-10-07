
find_package(Git QUIET)
set(GIT_STATUS "undefined")
if(GIT_FOUND)
	execute_process(
		COMMAND ${GIT_EXECUTABLE} log --pretty="%H" -n1
		OUTPUT_VARIABLE GIT_HASH
		RESULT_VARIABLE _GIT_LOG_RETURN
		OUTPUT_STRIP_TRAILING_WHITESPACE
		WORKING_DIRECTORY ${SPHYSICS_CMAKE_SOURCE_DIR}
		)
		if( _GIT_LOG_RETURN)
			unset(GIT_HASH)
		endif()

	execute_process(
		COMMAND ${GIT_EXECUTABLE} status -s
		OUTPUT_VARIABLE _GIT_STATUS
		RESULT_VARIABLE _GIT_LOG_RETURN
		OUTPUT_STRIP_TRAILING_WHITESPACE
		WORKING_DIRECTORY ${SPHYSICS_CMAKE_SOURCE_DIR}
		)
		if( _GIT_LOG_RETURN)
			unset(_GIT_STATUS)
		else()
			set(GIT_STATUS "${_GIT_STATUS}")
			unset(_GIT_STATUS)
		endif()
		string(REGEX REPLACE "\r?\n" "; " GIT_STATUS "${GIT_STATUS}")
endif()


execute_process(COMMAND hostname
	OUTPUT_VARIABLE HOSTNAME
	RESULT_VARIABLE _HOSTNAME_RETURN
	OUTPUT_STRIP_TRAILING_WHITESPACE
	)
if(_HOSTNAME_RETURN)
	unset(HOSTNAME)
endif()
unset(_HOSTNAME_RETURN)


set(USER $ENV{USER})


set(sphysics_VERSION "unknown")
message(STATUS "yy ${SPHYSICS_CMAKE_SOURCE_DIR}/src/sphysics")
execute_process(
COMMAND bash -c "grep '__version__ = '  _version.py"
RESULT_VARIABLE _sphysics_IMPORT_SUCCESS
OUTPUT_VARIABLE _sphysics_VERSION_RAW
ERROR_VARIABLE _sphysics_ERROR
OUTPUT_STRIP_TRAILING_WHITESPACE
WORKING_DIRECTORY ${SPHYSICS_CMAKE_SOURCE_DIR}/src/sphysics
)
if(_sphysics_IMPORT_SUCCESS EQUAL 0)
	# version extracted successfully
	set(sphysics_VERSION "${_sphysics_VERSION_RAW}")
	string(REGEX MATCH "'.*'" sphysics_VERSION "${sphysics_VERSION}")
	string(REGEX REPLACE "'" "" sphysics_VERSION "${sphysics_VERSION}")
endif()
unset(_sphysics_IMPORT_SUCCESS)
unset(_sphysics_VERSION_RAW)


configure_file(${SRC} ${DST} @ONLY)
