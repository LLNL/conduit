# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# Conduit Project Wide Includes
################################

# add conduit include dirs so units tests have access to the headers across
# libs and in unit tests

include_directories(${PROJECT_SOURCE_DIR}/libs/conduit 
                    ${PROJECT_SOURCE_DIR}/libs/conduit/c
                    ${PROJECT_BINARY_DIR}/libs/conduit)


include_directories(${PROJECT_SOURCE_DIR}/libs/relay
                    ${PROJECT_SOURCE_DIR}/libs/relay/c
                    ${PROJECT_BINARY_DIR}/libs/relay/)

include_directories(${PROJECT_SOURCE_DIR}/libs/blueprint
                    ${PROJECT_SOURCE_DIR}/libs/blueprint/c
                    ${PROJECT_BINARY_DIR}/libs/blueprint)


# Note: we use ENABLE_PYTHON instead of PYTHON_FOUND so this file 
# (SetupIncludes.cmake) can establish these paths before TPLs (Setup3rdParty.cmake)
# are setup. This shift solves a corner case constraining a conduit user.
if(ENABLE_PYTHON)
    include_directories(${PROJECT_SOURCE_DIR}/libs/conduit/python
                        ${PROJECT_BINARY_DIR}/libs/conduit/python
                        ${PROJECT_BINARY_DIR}/libs/relay/python
                        ${PROJECT_BINARY_DIR}/libs/blueprint/python)
endif()

