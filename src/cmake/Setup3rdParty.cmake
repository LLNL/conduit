###############################################################################
# Copyright (c) Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-666778
#
# All rights reserved.
#
# This file is part of Conduit.
#
# For details, see: http://software.llnl.gov/conduit/.
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


################################
# BLT provides support for:
#  gtest, fruit, and mpi
################################

if(UNIX AND NOT APPLE)
    # on some linux platforms we need to explicitly link threading
    # options.
    find_package( Threads REQUIRED )
endif()


################################
# Setup includes for RapidJSON
################################
include(cmake/thirdparty/SetupRapidJSON.cmake)
message(STATUS "Using RapidJSON Include: ${RAPIDJSON_INCLUDE_DIR}")
include_directories(${RAPIDJSON_INCLUDE_DIR})

################################
# Setup and build libb64
################################
add_subdirectory(thirdparty_builtin/libb64-1.2.1/)
include_directories(thirdparty_builtin/libb64-1.2.1/include/)

################################
# Setup and build libyaml
################################
add_subdirectory(thirdparty_builtin/libyaml-690a781/)
include_directories(thirdparty_builtin/libyaml-690a781/include)

################################
# Setup and build civetweb
################################
add_subdirectory(thirdparty_builtin/civetweb-0a95342/)
include_directories(thirdparty_builtin/civetweb-0a95342/include)

################################
# Setup includes for fmt
################################
include_directories(thirdparty_builtin/fmt-5.0.3/)


################################
# Optional Features
################################

if(ENABLE_PYTHON)
    ################################
    # Setup includes for Python & Numpy
    ################################
    include(cmake/thirdparty/SetupPython.cmake)
    message(STATUS "Using Python Include: ${PYTHON_INCLUDE_DIRS}")
    include_directories(${PYTHON_INCLUDE_DIRS})
    # if we don't find python, throw a fatal error
    if(NOT PYTHON_FOUND)
        message(FATAL_ERROR "ENABLE_PYTHON is true, but Python wasn't found.")
    endif()



    include(cmake/thirdparty/FindNumPy.cmake)
    message(STATUS "Using NumPy Include: ${NUMPY_INCLUDE_DIRS}")
    include_directories(${NUMPY_INCLUDE_DIRS})
    # if we don't find numpy, throw a fatal error
    if(NOT NUMPY_FOUND)
        message(FATAL_ERROR "ENABLE_PYTHON is true, but NumPy wasn't found.")
    endif()
endif()


################################
# Setup HDF5 if available
################################
# Search for HDF5.
if(HDF5_DIR)
    include(cmake/thirdparty/SetupHDF5.cmake)
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
    include(cmake/thirdparty/SetupSilo.cmake)
    include_directories(${SILO_INCLUDE_DIRS})
    # if we don't find silo, throw a fatal error
    if(NOT SILO_FOUND)
        message(FATAL_ERROR "SILO_DIR is set, but Silo wasn't found.")
    endif()
endif()

################################
# Setup ADIOS if available
################################
# Search for ADIOS.
if(ADIOS_DIR)
    include(cmake/thirdparty/SetupADIOS.cmake)
    include_directories(${ADIOS_INCLUDE_DIRS})
    # if we don't find ADIOS, throw a fatal error
    if(NOT ADIOS_FOUND)
        message(FATAL_ERROR "ADIOS_DIR is set, but ADIOS wasn't found.")
    endif()
endif()

################################
# Setup Zfp if available
################################
# Search for Zfp.
if(ZFP_DIR)
    include(cmake/thirdparty/SetupZfp.cmake)
    include_directories(${ZFP_INCLUDE_DIR})
    # if we don't find Zfp, throw a fatal error
    if(NOT ZFP_FOUND)
        message(FATAL_ERROR "ZFP_DIR is set, but Zfp wasn't found.")
    endif()
endif()
