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
# documentation targets
################################

# add a target to generate API documentation with Doxygen

add_custom_target(docs)

if(DOXYGEN_FOUND)
    add_subdirectory(doxygen)
endif()


if(SPHINX_FOUND)
    add_subdirectory(sphinx)
endif()


