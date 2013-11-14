################################
# Add our library
################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(conduit_sources Core.cpp Type.cpp)

add_library(conduit SHARED ${conduit_sources})
