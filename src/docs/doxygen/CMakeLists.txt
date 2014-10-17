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

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doxygen_docs
                 ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
                 WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                 COMMENT "Generating API documentation with Doxygen" VERBATIM)

add_dependencies(docs doxygen_docs)

install(CODE "execute_process(COMMAND ${CMAKE_BUILD_TOOL} doxygen_docs WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")")

install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html" 
        DESTINATION docs/doxygen/)
