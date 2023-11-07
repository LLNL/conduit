###############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other Ascent
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Ascent.
###############################################################################

###############################################################################
#
# Setup ZLIB_DIR
#
###############################################################################

if(ZLIB_DIR)
    set(ZLIB_ROOT ${ZLIB_DIR})
    find_package(ZLIB REQUIRED)
endif()

message(STATUS "ZLIB_LIBRARIES: ${ZLIB_LIBRARIES}")

# zlib dll
if(ZLIB_DIR)
    list(APPEND ASCENT_TPL_DLL_PATHS ${ZLIB_DIR}/bin/)
endif()