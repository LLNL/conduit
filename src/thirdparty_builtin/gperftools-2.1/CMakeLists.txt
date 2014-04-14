#
# Use CMake External Build to link google's perftools
#

include(ExternalProject)
ExternalProject_Add(
    gperftools_build
    PREFIX build
    URL ${CMAKE_CURRENT_SOURCE_DIR}/gperftools-2.1.tar.gz
    URL_MD5 5e5a981caf9baa9b4afe90a82dcf9882
    CONFIGURE_COMMAND ./configure --enable-frame-pointers --prefix=<INSTALL_DIR>
    BUILD_IN_SOURCE 1
)

