message_setup_this_dir()

set(_TESTS
    "math"
    "lorentz"
    "generators"
    "dirac"
    "tau_pwa"
    "kinematics"
    "variables"
    "utils"
    "polarimeter")

foreach(_TEST ${_TESTS})
  add_test(NAME ${_TEST} COMMAND ${Python_EXECUTABLE}
                                 ${CMAKE_CURRENT_SOURCE_DIR}/test_${_TEST}.py)
  set_property(
    TEST ${_TEST}
    APPEND
    PROPERTY ENVIRONMENT
             "PYTHONPATH=${Python_LIBRARY_OUTPUT_PATH}:$ENV{PYTHONPATH}"
             "SPHYSICS=${CMAKE_SOURCE_DIR}")
endforeach()
