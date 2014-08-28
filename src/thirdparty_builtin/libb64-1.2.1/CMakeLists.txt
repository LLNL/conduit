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
# Simple CMake setup for libb64
################################

#
# libb64 sources
#

set(libb64_sources
    src/cdecode.c	
    src/cencode.c
    )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(libb64 STATIC ${libb64_sources} )

