#############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC
# Produced at the Lawrence Livermore National Laboratory. 
# 
# All rights reserved.
# 
# This source code cannot be distributed without further review from 
# Lawrence Livermore National Laboratory.
#############################################################################

cmake_minimum_required(VERSION 2.8)

################################
# Conduit
################################

project(conduit)

################################
# Build Options
################################
option(ENABLE_GPREF_TOOLS "Build with google perftools" OFF)
option(ENABLE_PYTHON      "Build Python Support"        OFF)
option(ENABLE_MPI         "Build MPI Support"           OFF)
option(ENABLE_SILO        "Build Silo Support"          OFF)


################################
# Set some standard cmake opts
################################
include(CMake/CMakeBasics.cmake)

################################
# Checks for type sizes, etc
################################
include(CMake/BasicChecks.cmake)

################################
# Setup 3rd Party Libs
################################
include(CMake/Setup3rdParty.cmake)

################################
# Project Wide Includes
################################
# add conduit as an include dirs so units tests have access to the headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/conduit 
                    ${CMAKE_CURRENT_BINARY_DIR}/conduit )


################################
# Add the main lib
################################
add_subdirectory(conduit)

################################
# Add the conduit-silo lib 
################################
if(SILO_FOUND)
    add_subdirectory(conduit_silo)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/conduit_silo 
                        ${CMAKE_CURRENT_BINARY_DIR}/conduit_silo )
endif()

################################
# Add docs
################################
add_subdirectory(docs)

################################
# Add our tests
################################
add_subdirectory(tests)

