################################
# Conduit
################################

cmake_minimum_required(VERSION 2.8)

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
# Setup and build libb64
################################
add_subdirectory(thirdparty_builtin/libb64-1.2.1/)
include_directories(thirdparty_builtin/libb64-1.2.1/include/)


################################
# Setup and build gperftools
################################
add_subdirectory(thirdparty_builtin/gperftools-2.1/)

add_library(gperftools_lib STATIC IMPORTED)

set_target_properties(gperftools_lib PROPERTIES IMPORTED_LOCATION 
                      ${CMAKE_BINARY_DIR}/thirdparty_builtin/gperftools-2.1/build/lib/libtcmalloc_and_profiler.a)

add_dependencies( gperftools_lib gperftools_build )

include_directories(${CMAKE_BINARY_DIR}/thirdparty_builtin/gperftools-2.1/build/include/)

#
# Note: We only want to do this when are using gperf profiling tools, we may not want to use this in general
#
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-fno-omit-frame-pointer") 
endif()


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
