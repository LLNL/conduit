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
# For details, see https://lc.llnl.gov/conduit/.
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
#
# Example that shows how to use an installed instance of Conduit in another
# CMake-based build system.
#
# To build:
#  mkdir build
#  cd build
#  cmake -DCONDUIT_DIR={conduit install path} ../
#  make
#  ./conduit_example
#
#
# If run in sub directory of a conduit install, 
# CONDUIT_DIR will default to ../../..
# 
#  mkdir build
#  cd build
#  cmake ..
#  make
#  ./conduit_example
#
###############################################################################

cmake_minimum_required(VERSION 3.0)

project(using_with_cmake)


################################################################
# Option 1: import conduit using an explicit path (recommended)
################################################################

#
# Provide default CONDUIT_DIR that works in a conduit install
#
if(NOT CONDUIT_DIR)
    set(CONDUIT_DIR "../../..")
endif()

#
# Check for valid conduit install
#

# if given relative path, resolve
get_filename_component(CONDUIT_DIR ${CONDUIT_DIR} ABSOLUTE)
# check for expected cmake exported target files
if(NOT EXISTS ${CONDUIT_DIR}/lib/cmake/conduit/ConduitConfig.cmake)
    # also allow old style, but if neither case is found display error msg
    if(NOT EXISTS ${CONDUIT_DIR}/lib/cmake/ConduitConfig.cmake)
        MESSAGE(FATAL_ERROR "Could not find Conduit CMake include file (${CONDUIT_DIR}/lib/cmake/conduit/ConduitConfig.cmake)")
    endif()
endif()

#
# Use CMake's find_package to import conduit's targets
# using explicit path
#
find_package(Conduit REQUIRED
             NO_DEFAULT_PATH 
             PATHS ${CONDUIT_DIR}/lib/cmake/conduit)


################################################################
# Option 2: import conduit using find_package search
################################################################
##
## Add Conduit's install path to CMAKE_PREFIX_PATH 
## and use following find_package call to import conduit's targets
##
#
# find_package(Conduit REQUIRED)
#


#######################################################
# create our example 
#######################################################
add_executable(conduit_example conduit_example.cpp)

# link to conduit targets
target_link_libraries(conduit_example conduit::conduit)

# if you are using conduit's python CAPI capsule interface
# target_link_libraries(conduit_example conduit::conduit_python)


# if you are using conduit's mpi features
# target_link_libraries(conduit_example conduit::conduit_mpi)


