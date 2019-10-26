###############################################################################
# Copyright (c) Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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

cmake_minimum_required(VERSION 3.0)
################################
# cmake policy selections
################################
# cmake var deref policy
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 OLD)
endif()

# allow find_packages to use ZZZ_ROOT vars
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

################################
# Conduit
################################

project(conduit VERSION "0.5.0")

################################
# Build Options
################################
option(BUILD_SHARED_LIBS  "Build shared libraries"      ON)
option(ENABLE_TESTS       "Build conduit tests"         ON)
option(ENABLE_EXAMPLES    "Build Examples"              ON)
option(ENABLE_UTILS       "Build Utilities"             ON)
option(ENABLE_DOCS        "Build conduit documentation" ON)
option(ENABLE_COVERAGE    "Build with coverage flags"   OFF)

option(ENABLE_PYTHON      "Build Python Support"        OFF)
option(ENABLE_FORTRAN     "Build Fortran  support"      OFF)

option(ENABLE_MPI         "Build MPI Support"           OFF)

# Add another option that provides extra 
# control over conduit tests for cases where 
# conduit is brought in as a submodule
option(CONDUIT_ENABLE_TESTS "Build conduit tests" ON)

################################
# Invoke CMake Fortran setup
# if ENABLE_FORTRAN == ON
################################
if(ENABLE_FORTRAN)
    enable_language(Fortran)
endif()

# don't use BLT's all warnings feature
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "")

################################
# Init BLT
################################
# This also includes 
# Conduit's BLT defaults
include(cmake/SetupBLT.cmake)

################################
# Set some standard cmake opts
################################
include(cmake/CMakeBasics.cmake)

################################
# Fortran Support
################################
include(cmake/SetupFortran.cmake)

################################
# Checks for type sizes, etc
################################
include(cmake/BasicTypeChecks.cmake)

################################
# Setup project wide includes
################################
include(cmake/SetupIncludes.cmake)

################################
# Setup 3rd Party Libs
################################
include(cmake/Setup3rdParty.cmake)

################################
# Setup tests helpers
################################
include(cmake/SetupTests.cmake)

################################
# Add our libs
################################
add_subdirectory(libs)

################################
# Add our examples
################################
add_subdirectory(examples)

################################
# Add docs
################################
if(ENABLE_DOCS)
    add_subdirectory(docs)
else()
    message(STATUS "Skipping documentation targets (ENABLE_DOCS = OFF)")
endif()

################################
# Add our tests
################################
if(ENABLE_TESTS)
    if(CONDUIT_ENABLE_TESTS)
        add_subdirectory(tests)
    else()
        message(STATUS "Skipping test targets (CONDUIT_ENABLE_TESTS = OFF)")
    endif()
else()
    message(STATUS "Skipping test targets (ENABLE_TESTS = OFF)")
endif()

################################
# Add our config helpers
################################
add_subdirectory(config)

################################
# Create CMake importable
# exports for all of our targets
################################
install(EXPORT conduit DESTINATION lib/cmake)
