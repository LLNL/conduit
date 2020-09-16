# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

####################################
# Simple CMake setup for civetweb
####################################

#
# civetweb sources
#


set(civetweb_headers
    include/civetweb.h
    include/CivetServer.h
    )

set(civetweb_sources
    src/civetweb.c
    src/CivetServer.cpp
    )

add_definitions(-DUSE_WEBSOCKET)



############################################
# disable ssl support for the static case
############################################
# We don't want to rely on dlopen NO_SSL_DL and we don't require a
# openssl install.
if(NOT BUILD_SHARED_LIBS)
    add_definitions(-DNO_SSL_DL -DNO_SSL)
endif()


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)


add_compiled_library(NAME   conduit_civetweb
                     OBJECT
                     HEADERS ${civetweb_headers}
                     SOURCES ${civetweb_sources}
                     FOLDER thirdparty_builtin)

#  avoid unknown pragma warnings when compiling civetweb with clang
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    add_target_compile_flags(TARGET conduit_civetweb
                             FLAGS -Wno-unknown-pragmas)
endif()


