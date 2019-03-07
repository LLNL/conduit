###############################################################################
# Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-666778
#
# All rights reserved.
#
# This file is part of Conduit.
#
# For details, see: http://software.llnl.gov/conduit/.
#
# Please also read conduit/LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

# Configure our Makefile-based build helper
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_config.mk.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_config.mk")

if (NOT DEFINED CONDUIT_INSTALL_INCLUDE_DIR)
  set(CONDUIT_INSTALL_INCLUDE_DIR "include")
endif()

if (NOT DEFINED CONDUIT_INSTALL_CONFIG_DIR)
  set(CONDUIT_INSTALL_CONFIG_DIR "lib")
endif()

if (NOT DEFINED CONDUIT_INSTALL_LIB_DIR)
  set(CONDUIT_INSTALL_LIB_DIR "lib")
endif()

if (NOT DEFINED CONDUIT_INSTALL_BIN_DIR)
  set(CONDUIT_INSTALL_BIN_DIR "bin")
endif()

if (NOT DEFINED CONDUIT_INSTALL_SHARED_RESOURCES_DIR)
  set(CONDUIT_INSTALL_SHARED_RESOURCES_DIR "share/conduit")
endif()

if (NOT DEFINED CONDUIT_INSTALL_CMAKE_MODULE_DIR)
  set(CONDUIT_INSTALL_CMAKE_MODULE_DIR "${CONDUIT_INSTALL_CONFIG_DIR}/cmake")
endif()

include(CMakePackageConfigHelpers)
# setup cmake exports
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/conduit-config.cmake.in
  ${PROJECT_BINARY_DIR}/${CONDUIT_INSTALL_CMAKE_MODULE_DIR}/conduit-config.cmake
  INSTALL_DESTINATION ${CONDUIT_INSTALL_CONFIG_DIR}
  PATH_VARS
    CONDUIT_INSTALL_INCLUDE_DIR
    CONDUIT_INSTALL_LIB_DIR
    CONDUIT_INSTALL_BIN_DIR
    CONDUIT_INSTALL_SHARED_RESOURCES_DIR
    CONDUIT_INSTALL_CMAKE_MODULE_DIR
    )

configure_file (${PROJECT_SOURCE_DIR}/cmake/conduit-deps-paths.cmake.in
                ${PROJECT_BINARY_DIR}/${CONDUIT_INSTALL_CMAKE_MODULE_DIR}/conduit-deps-paths.cmake)

# install extra cmake files
install(FILES
        ${PROJECT_BINARY_DIR}/${CONDUIT_INSTALL_CMAKE_MODULE_DIR}/conduit-config.cmake
        ${PROJECT_BINARY_DIR}/${CONDUIT_INSTALL_CMAKE_MODULE_DIR}/conduit-deps-paths.cmake
        DESTINATION ${CONDUIT_INSTALL_CMAKE_MODULE_DIR})

# install our config.mk
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/conduit_config.mk"
        DESTINATION share/conduit/)

