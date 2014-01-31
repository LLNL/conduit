################################
# Conduit
################################

cmake_minimum_required(VERSION 2.6)

project(conduit)

################################
# Checks for type sizes
################################

INCLUDE (CheckTypeSize)
CHECK_TYPE_SIZE("boolean"  SIZEOF_BOOLEAN)
CHECK_TYPE_SIZE("char"     SIZEOF_CHAR)
CHECK_TYPE_SIZE("short"    SIZEOF_SHORT)
CHECK_TYPE_SIZE("int"      SIZEOF_INT)
CHECK_TYPE_SIZE("long"     SIZEOF_LONG)

CHECK_TYPE_SIZE("float"    SIZEOF_FLOAT)
CHECK_TYPE_SIZE("double"   SIZEOF_DOUBLE)

CHECK_TYPE_SIZE("long long"     SIZEOF_LONG_LONG)
CHECK_TYPE_SIZE("long float"    SIZEOF_LONG_FLOAT)
CHECK_TYPE_SIZE("long double"   SIZEOF_LONG_DOUBLE)

CHECK_TYPE_SIZE("void *"        SIZEOF_VOID_P)


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
# 
# # we will need to get the proper Python
# include(CMake/FindPython.cmake)
# message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
# include_directories(${PYTHON_INCLUDE_DIRS})
# 
# 
# 
# include(CMake/FindNumPy.cmake)
# message(STATUS "Using NumPy Include: ${NUMPY_INCLUDE_DIRS}")
# include_directories(${NUMPY_INCLUDE_DIRS})


################################
# Setup MPI if available 
################################
# Search for MPI.
include(FindMPI)


################################
# Project Wide Includes
################################
# add conduit as an include dir as well so units tests can use it
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/conduit 
                    ${CMAKE_CURRENT_BINARY_DIR}/conduit )


################################
# Add doxygen
################################
add_subdirectory(doxy)

################################
# Add the main lib
################################
add_subdirectory(conduit)

################################
# Add our unit tests
################################
add_subdirectory(tests)
