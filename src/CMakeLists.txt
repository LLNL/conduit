################################
# Conduit
################################

cmake_minimum_required(VERSION 2.6)

project(conduit)


################################
# Enable GTest
################################
add_subdirectory(thirdparty_builtin/gtest-1.7.0)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Setup includes for RapidJSON
################################
include(CMake/FindRapidJSON.cmake)
message(STATUS "Using RapidJSON Include: ${RAPIDJSON_INCLUDE_DIR}")
include_directories(${RAPIDJSON_INCLUDE_DIR})


################################
# Setup includes for Numpy
################################

# we will need to get the proper Python
include(CMake/FindPython.cmake)
message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
include_directories(${PYTHON_INCLUDE_DIRS})



include(CMake/FindNumPy.cmake)
message(STATUS "Using NumPy Include: ${NUMPY_INCLUDE_DIRS}")
include_directories(${NUMPY_INCLUDE_DIRS})



# add conduit as an include dir as well so units tests can use it
include_directories(conduit)


################################
# Add the main lib
################################
add_subdirectory(conduit)

################################
# Add our unit tests
################################
add_subdirectory(tests)
