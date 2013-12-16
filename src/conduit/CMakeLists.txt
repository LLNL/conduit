################################
# Add our library
################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(conduit_sources
    conduit.h
    Core.cpp
    Core.h 
    Endianness.cpp 
    Endianness.h
    Type.cpp
    Type.h
    Node.cpp
    Node.h
    Array.cpp
    Array.h
    GhostNode.cpp
    GhostNode.h)

add_library(conduit SHARED ${conduit_sources})
