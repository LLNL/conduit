#############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC
# Produced at the Lawrence Livermore National Laboratory. 
# 
# All rights reserved.
# 
# This source code cannot be distributed without further review from 
# Lawrence Livermore National Laboratory.
#############################################################################

################################
# Conduit 3rd Party Dependencies
################################

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
# Optional Features
################################

################################
# Documentation Packages
################################

find_package(Doxygen)
include(CMake/FindSphinx.cmake)


if(ENABLE_GPERFTOOLS)
    ################################
    # Setup and build gperftools
    ################################
    set(GPREFTOOLS_DIR thirdparty_builtin/gperftools-2.2.1)
    add_subdirectory(${GPREFTOOLS_DIR})
    add_library(gperftools_lib STATIC IMPORTED)

    set_target_properties(gperftools_lib PROPERTIES IMPORTED_LOCATION 
                      ${CMAKE_BINARY_DIR}/${GPREFTOOLS_DIR}/build/lib/libtcmalloc_and_profiler.a)

    add_dependencies( gperftools_lib gperftools_build )

    include_directories(${CMAKE_BINARY_DIR}/${GPREFTOOLS_DIR}/build/include/)

    #
    # Note: We only want to do this when are using gperf profiling tools, 
    # we may not want to use this in general
    #
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(CMAKE_CXX_FLAGS "-fno-omit-frame-pointer") 
    endif()
endif()

if(ENABLE_PYTHON)
    ################################
    # Setup includes for Python & Numpy
    ################################
    include(CMake/FindPython.cmake)
    message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    
    include(CMake/FindNumPy.cmake)
    message(STATUS "Using NumPy Include: ${NUMPY_INCLUDE_DIRS}")
    include_directories(${NUMPY_INCLUDE_DIRS})
endif()

################################
# Setup MPI if available 
################################
# Search for MPI.
if(ENABLE_MPI)
    include(FindMPI)
endif()
