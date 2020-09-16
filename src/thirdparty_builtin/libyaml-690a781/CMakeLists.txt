# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# Simple CMake setup for libyaml
################################

#
# libyaml sources
#

set(libyaml_sources
    src/api.c
    src/dumper.c
    src/emitter.c
    src/loader.c
    src/parser.c
    src/reader.c
    src/scanner.c
    src/writer.c
    )

#
# libyaml headers
#

set(libyaml_headers
    include/yaml.h
    src/yaml_private.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_compiled_library(NAME conduit_libyaml
                     OBJECT
                     SOURCES ${libyaml_sources}
                     HEADERS ${libyaml_headers}
                     FOLDER thirdparty_builtin)

# use cxx linker to avoid msvc static build issues
set_target_properties(conduit_libyaml PROPERTIES LINKER_LANGUAGE CXX)

if(WIN32)
    blt_add_target_definitions(TO conduit_libyaml
                               SCOPE PRIVATE
                               TARGET_DEFINITIONS CONDUIT_PLATFORM_WINDOWS)
endif()

