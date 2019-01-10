###############################################################################
# Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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

include("cmake/BitwidthMapping.cmake")


convert_to_native_escaped_file_path(${CMAKE_INSTALL_PREFIX}
                                    CONDUIT_INSTALL_PREFIX)

set(CONDUIT_INSTALL_PREFIX ${CONDUIT_INSTALL_PREFIX} CACHE STRING "" FORCE)

if(FORTRAN_FOUND)
    set(CONDUIT_FORTRAN_COMPILER ${CMAKE_Fortran_COMPILER})
endif()


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_config.h")

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_bitwidth_style_types.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_bitwidth_style_types.h")

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_exports.h")

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
    conduit_log.hpp
    conduit_utils.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_exports.h
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_config.h
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_bitwidth_style_types.h
    )

#
# Specify conduit c interface headers
#
set(conduit_c_headers
    c/conduit.h
    c/conduit_datatype.h
    c/conduit_node.h
    c/conduit_cpp_to_c.hpp
    c/conduit_utils.h
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
    conduit_log.cpp
    conduit_utils.cpp
    )

#
# Specify conduit c interface sources
#
set(conduit_c_sources
    c/conduit_c.cpp
    c/conduit_datatype_c.cpp
    c/conduit_node_c.cpp
    c/conduit_cpp_to_c.cpp
    c/conduit_utils_c.cpp
    )



#################################
# Add fortran interface lib
# if fortran support was selected
#################################
if(FORTRAN_FOUND)
    # Create include file with bitwidth style mappings for fortran
    configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/fortran/conduit_fortran_bitwidth_style_types.inc.in"
                    "conduit_fortran_bitwidth_style_types.inc")

    # Allow fortran compiler to include the Bitwidth_Style_Types.inc file
    # Note: Unlike .h files, the .inc file will not be needed by client code
    include_directories(${CMAKE_CURRENT_BINARY_DIR})

    #
    # Specify fortran sources
    #
    set(conduit_fortran_sources
        fortran/conduit_fortran.F90)

    #  add oo interface if the fortran compiler supports it
    if(ENABLE_FORTRAN_OBJ_INTERFACE)
        list(APPEND conduit_fortran_sources fortran/conduit_fortran_obj.f90)
    endif()

endif()

#
# Setup the conduit lib
#
add_compiled_library(NAME   conduit
                     EXPORT conduit
                     HEADERS ${conduit_headers} ${conduit_c_headers}
                     SOURCES ${conduit_sources} ${conduit_c_sources} ${conduit_fortran_sources}
                             $<TARGET_OBJECTS:conduit_b64>
                     HEADERS_DEST_DIR include/conduit)


#################################
# Fortran related target options
#################################
if(FORTRAN_FOUND)
    if("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
         # disable c <-> fortran binding warnings for gfortran 
         message(STATUS "Disabling c <-> fortran binding warnings for "
                         "conduit "
                         "(Adding Fortran compiler flag: -Wno-c-binding-type)")

        set_source_files_properties(${conduit_fortran_sources}
                                    PROPERTIES COMPILE_FLAGS "-Wno-c-binding-type")
    endif()
endif()

################################
# Add python wrappers if python
# support was selected
################################
if(PYTHON_FOUND)
    add_subdirectory(python)
endif()


####################################################################
# Special install targets for conduit fortran modules 
####################################################################
if(FORTRAN_FOUND)
    set(conduit_fortran_modules
        ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit.mod)

    if(ENABLE_FORTRAN_OBJ_INTERFACE)
        list(APPEND conduit_fortran_modules
                    ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit_obj.mod)
    endif()

    # Setup install to copy the fortran modules 
    install(FILES 
            ${conduit_fortran_modules}
            DESTINATION include/conduit)
endif()


