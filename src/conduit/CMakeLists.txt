################################
# Add our library
################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_library(conduit SHARED conduit.cpp)
