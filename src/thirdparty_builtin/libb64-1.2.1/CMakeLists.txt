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

