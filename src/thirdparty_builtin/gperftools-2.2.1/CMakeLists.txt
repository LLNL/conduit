#
# Use CMake External Build to link google's perftools
#

include(ExternalProject)
ExternalProject_Add(
    gperftools_build
    PREFIX build
    URL ${CMAKE_CURRENT_SOURCE_DIR}/gperftools-2.2.1.tar.gz
    URL_MD5 47cbede9442764f9e8fbc403af6cacfd
    CONFIGURE_COMMAND  CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} ./configure 
                        --enable-heap-checker     
                        --enable-heap-profiler 
                        --enable-cpu-profiler
                        --enable-frame-pointers 
                        --prefix=<INSTALL_DIR>
    BUILD_IN_SOURCE 1
)

