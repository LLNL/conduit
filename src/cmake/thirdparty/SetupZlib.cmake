# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
###############################################################################
#

###############################################################################
#
# Setup ZLIB
#
###############################################################################

if(ZLIB_DIR)
    set(ZLIB_ROOT ${ZLIB_DIR})
    find_package(ZLIB REQUIRED)
endif()

message(STATUS "ZLIB_LIBRARIES: ${ZLIB_LIBRARIES}")
