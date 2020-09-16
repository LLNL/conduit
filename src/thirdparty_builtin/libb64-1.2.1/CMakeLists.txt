# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# Simple CMake setup for libb64
################################

#
# libb64 sources
#

set(libb64_sources
    src/cdecode.cpp
    src/cencode.cpp
    )

set(libb64_headers
    include/b64/cdecode.h
    include/b64/cencode.h
    include/b64/decode.h
    include/b64/encode.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


add_compiled_library(NAME conduit_b64
                     OBJECT
                     SOURCES ${libb64_sources}
                     HEADERS ${libb64_headers}
                     FOLDER thirdparty_builtin)

