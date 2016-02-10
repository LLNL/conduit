###############################################################################
# Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://llnl.github.io/conduit/.
# 
# Please also read conduit/LICENSE
# 
# Redistribution and use in source and binary forms, with or without 
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, 
#   this list of conditions and the disclaimer below.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
# 
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
# POSSIBILITY OF SUCH DAMAGE.
# 
###############################################################################

################################
# Conduit 3rd Party Dependencies
################################

if(ENABLE_TESTS)
    ################################
    # Enable GTest
    ################################
    
    #
    # We always want to build gtest as a static lib, however
    # it shares our "BUILD_SHARED_LIBS" option, so we need
    # to force this value to OFF, and then restore the 
    # previous setting.
    #

    set(BSL_ORIG_VALUE ${BUILD_SHARED_LIBS})
    
    set(BUILD_SHARED_LIBS OFF)
    add_subdirectory(thirdparty_builtin/gtest-1.7.0)
    
    set(BUILD_SHARED_LIBS ${BSL_ORIG_VALUE})
    
    enable_testing()
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
endif()

################################
# Setup includes for RapidJSON
################################
include(CMake/thirdparty/FindRapidJSON.cmake)
message(STATUS "Using RapidJSON Include: ${RAPIDJSON_INCLUDE_DIR}")
include_directories(${RAPIDJSON_INCLUDE_DIR})

################################
# Setup and build libb64
################################
add_subdirectory(thirdparty_builtin/libb64-1.2.1/)
include_directories(thirdparty_builtin/libb64-1.2.1/include/)

################################
# Setup and build civetweb
################################
add_subdirectory(thirdparty_builtin/civetweb/)
include_directories(thirdparty_builtin/civetweb/include)

################################
# Optional Features
################################

################################
# Documentation Packages
################################

find_package(Doxygen)
include(CMake/thirdparty/FindSphinx.cmake)


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
    include(CMake/thirdparty/FindPython.cmake)
    message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    # if we don't find python, throw a fatal error
    if(NOT PYTHON_FOUND)
        message(FATAL_ERROR "ENABLE_PYTHON is true, but Python wasn't found.")
    endif()


    
    include(CMake/thirdparty/FindNumPy.cmake)
    message(STATUS "Using NumPy Include: ${NUMPY_INCLUDE_DIRS}")
    include_directories(${NUMPY_INCLUDE_DIRS})
    # if we don't find numpy, throw a fatal error
    if(NOT NUMPY_FOUND)
        message(FATAL_ERROR "ENABLE_PYTHON is true, but NumPy wasn't found.")
    endif()
endif()

################################
# Setup MPI if available 
################################
# Search for MPI.
if(ENABLE_MPI)
    include(FindMPI)
    # if we don't find mpi, throw a fatal error
    if(NOT MPI_FOUND)
        message(FATAL_ERROR "ENABLE_MPI is true, but MPI wasn't found.")
    endif()
endif()


################################
# Setup HDF5 if available 
################################
# Search for HDF5.
if(HDF5_DIR)
    include(CMake/thirdparty/FindHDF5.cmake)
    include_directories(${HDF5_INCLUDE_DIRS})
    # if we don't find HDF5, throw a fatal error
    if(NOT HDF5_FOUND)
        message(FATAL_ERROR "HDF5_DIR is set, but HDF5 wasn't found.")
    endif()
endif()

################################
# Setup Silo if available 
################################
# Search for Silo.
if(SILO_DIR)
    include(CMake/thirdparty/FindSilo.cmake)
    include_directories(${SILO_INCLUDE_DIRS})
    # if we don't find silo, throw a fatal error
    if(NOT SILO_FOUND)
        message(FATAL_ERROR "SILO_DIR is set, but Silo wasn't found.")
    endif()
endif()

################################
# Setup fruit (fortran uint testing framework) if fortran is enabled
################################
if(ENABLE_FORTRAN)
    add_subdirectory(thirdparty_builtin/fruit-3.3.9)
endif()

