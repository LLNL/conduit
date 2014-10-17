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

# add a target to generate documentation with Sphinx


# configured documentation tools and intermediate build results
set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

# Sphinx cache with pickled ReST documents
set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

# HTML output directory
set(SPHINX_HTML_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
               "${BINARY_BUILD_DIR}/conf.py"
               @ONLY)

add_custom_target(sphinx_docs 
        ${SPHINX_EXECUTABLE}
        -q -b html
        -W
        -c "${BINARY_BUILD_DIR}"
        -d "${SPHINX_CACHE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${SPHINX_HTML_DIR}"
        COMMENT "Building HTML documentation with Sphinx"
        DEPENDS doxygen_docs)
        

add_dependencies(docs sphinx_docs)

######
# This snippet makes sure if we do a make install w/o the optional "docs"
# target built, it will be built during the install process.
######


install(CODE "execute_process(COMMAND ${CMAKE_BUILD_TOOL} sphinx_docs WORKING_DIRECTORY \"${CMAKE_CURRENT_BINARY_DIR}\")")

install(DIRECTORY "${SPHINX_HTML_DIR}" 
        DESTINATION docs/sphinx/)