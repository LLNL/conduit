###############################################################################
# Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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


################################
# Add our library
################################

ENABLE_WARNINGS()

include("CMake/BitwidthChecks.cmake")


set(CONDUIT_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_config.h")

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_bitwidth_style_types.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_bitwidth_style_types.h")

#
# Specify conduit cpp headers
#
set(conduit_headers
    conduit.hpp
    conduit_endianness_types.h
    conduit_core.hpp
    conduit_endianness.hpp
    conduit_data_array.hpp
    conduit_data_type.hpp
    conduit_node.hpp
    conduit_generator.hpp
    conduit_error.hpp
    conduit_node_iterator.hpp
    conduit_schema.hpp
    conduit_utils.hpp
    conduit_exports.h
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_config.h
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_bitwidth_style_types.h
    )

#
# Specify conduit c interface headers
#
set(conduit_c_headers
    c/conduit.h
    c/conduit_node.h
    c/conduit_cpp_to_c.hpp
    )

#
# Specify conduit cpp sources
#
set(conduit_sources
    conduit_core.cpp
    conduit_error.cpp
    conduit_endianness.cpp
    conduit_data_type.cpp
    conduit_data_array.cpp
    conduit_generator.cpp
    conduit_node.cpp
    conduit_node_iterator.cpp
    conduit_schema.cpp
    conduit_utils.cpp
    )

#
# Specify conduit c interface sources
#
set(conduit_c_sources
    c/conduit_c.cpp
    c/conduit_node_c.cpp
    c/conduit_cpp_to_c.cpp
    )


#
# Setup the conduit lib
#
add_compiled_library(NAME   conduit
                     EXPORT conduit
                     HEADERS ${conduit_headers} ${conduit_c_headers}
                     SOURCES ${conduit_sources} ${conduit_c_sources}
                             $<TARGET_OBJECTS:conduit_b64>
                     HEADERS_DEST_DIR include/conduit)

################################
# Add python wrappers if python
# support was selected
################################
if(PYTHON_FOUND)
    add_subdirectory(python)
endif()

#################################
# Add fortran interface lib
# if fortran support was selected
#################################
if(FORTRAN_FOUND)
    add_subdirectory(fortran)
endif()
