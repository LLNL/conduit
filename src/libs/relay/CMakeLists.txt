###############################################################################
# Copyright (c) Copyright (c) 2015-2018, Lawrence Livermore National Security, LLC.
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
# Add the conduit relay library
################################

ENABLE_WARNINGS()

####
# enable optional features if deps exists
####

if(HDF5_FOUND)
  SET(CONDUIT_RELAY_IO_HDF5_ENABLED TRUE)
endif()

if(SILO_FOUND)
  SET(CONDUIT_RELAY_IO_SILO_ENABLED TRUE)
endif()

if(ADIOS_FOUND)
  SET(CONDUIT_RELAY_IO_ADIOS_ENABLED TRUE)
endif()

if(MPI_FOUND)
  SET(CONDUIT_RELAY_MPI_ENABLED TRUE)
endif()


convert_to_native_escaped_file_path(${CMAKE_CURRENT_SOURCE_DIR}
                                    CONDUIT_RELAY_SOURCE_DIR)


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_relay_config.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_config.h")

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_relay_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_exports.h")


#
# Specify cpp headers
#
set(conduit_relay_headers
    conduit_relay.hpp
    conduit_relay_io.hpp
    conduit_relay_io_identify_protocol.hpp
    conduit_relay_io_blueprint.hpp
    conduit_relay_web.hpp
    conduit_relay_web_node_viewer_server.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_exports.h
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_config.h)

#
# Specify c headers
#
set(conduit_relay_c_headers
    c/conduit_relay.h
    c/conduit_relay_io.h)

#
# Specify cpp sources
#
set(conduit_relay_sources
    conduit_relay.cpp
    conduit_relay_io.cpp
    conduit_relay_io_identify_protocol.cpp
    conduit_relay_io_blueprint.cpp
    conduit_relay_web.cpp
    conduit_relay_web_node_viewer_server.cpp)

#
# Specify c sources
#
set(conduit_relay_c_sources
    c/conduit_relay_c.cpp
    c/conduit_relay_io_c.cpp)


if(HDF5_FOUND)
    list(APPEND conduit_relay_headers
         conduit_relay_hdf5.hpp 
         conduit_relay_io_hdf5.hpp
         conduit_relay_io_hdf5_api.hpp
        )
    list(APPEND conduit_relay_sources conduit_relay_io_hdf5.cpp)
endif()

if(SILO_FOUND)
    list(APPEND conduit_relay_headers 
         conduit_relay_silo.hpp 
         conduit_relay_io_silo.hpp
         conduit_relay_io_silo_api.hpp
        )
    list(APPEND conduit_relay_sources conduit_relay_io_silo.cpp)
endif()

if(ADIOS_FOUND)
    list(APPEND conduit_relay_headers
         conduit_relay_io_adios.hpp
         conduit_relay_io_adios_api.hpp
        )
    list(APPEND conduit_relay_sources conduit_relay_io_adios.cpp)
endif()

################################
# Add fortran interface if 
# fortran support was selected
################################
#
# Specify fortran sources
#
set(conduit_relay_fortran_sources
    fortran/conduit_relay_fortran.f)

#
# setup conduit relay lib
#


##################################
# system dependencies for civetweb
##################################


#
# Link with conduit
#
set(conduit_relay_deps conduit conduit_blueprint)

if(UNIX AND NOT APPLE)
    # we need these for civetweb on linux, we may need similar libs
    # on windows (OSX appears ok without them)
    list(APPEND conduit_relay_deps dl rt ${CMAKE_THREAD_LIBS_INIT})
endif()

#
# Link with silo and hdf5 and adios if these are enabled
#
if(SILO_FOUND)
    list(APPEND conduit_relay_deps silo)
endif()

if(HDF5_FOUND)
    list(APPEND conduit_relay_deps hdf5)
    if(HDF5_IS_PARALLEL AND MPI_FOUND)
        list(APPEND conduit_relay_deps mpi)
    endif()
endif()

if(ADIOS_FOUND)
    # Link with the serial ADIOS libraries.
    list(APPEND conduit_relay_deps adios_nompi)
endif()


add_compiled_library(NAME conduit_relay
                     EXPORT conduit
                     HEADERS ${conduit_relay_headers} ${conduit_relay_c_headers}
                     SOURCES ${conduit_relay_sources} ${conduit_relay_c_sources}
                             ${conduit_relay_fortran_sources}
                             $<TARGET_OBJECTS:conduit_civetweb>
                     DEPENDS_ON ${conduit_relay_deps}
                     HEADERS_DEST_DIR include/conduit)

if(FORTRAN_FOUND)
    set_target_properties(conduit_relay
                          PROPERTIES Fortran_FORMAT "FREE")
endif()



###################################
# add conduit_relay_node_viewer exe
###################################

blt_add_executable(
    NAME        conduit_relay_node_viewer
    SOURCES     conduit_relay_web_node_viewer_exe.cpp
    DEPENDS_ON  conduit_relay
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# add install target for conduit_relay_node_viewer
install(TARGETS conduit_relay_node_viewer
        RUNTIME DESTINATION bin)


###################################
# add conduit_relay_io_ls exe
###################################
blt_add_executable(
    NAME        conduit_relay_io_ls
    SOURCES     conduit_relay_io_ls_exe.cpp
    DEPENDS_ON  conduit_relay
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})


# add install target
install(TARGETS conduit_relay_io_ls
        RUNTIME DESTINATION bin)

###################################
# add conduit_relay_io_convert exe
###################################

blt_add_executable(
    NAME        conduit_relay_io_convert
    SOURCES     conduit_relay_io_convert_exe.cpp
    DEPENDS_ON  conduit_relay
    OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# add install target for conduit_relay_io_convert
install(TARGETS conduit_relay_io_convert
        RUNTIME DESTINATION bin)


##############################################################
# install files needed for the relay web node viewer client
#############################################################
install(DIRECTORY web_clients/node_viewer
        DESTINATION share/conduit/web_clients/)

#############################################################
# add install of entangle helper script
#############################################################
install(FILES scripts/conduit_relay_entangle.py
        DESTINATION bin)


################################
# Add python wrappers if python
# support was selected
################################
if(PYTHON_FOUND)
    add_subdirectory(python)
endif()


###############################################################################
# Special install targets for conduit fortran modules 
###############################################################################
if(FORTRAN_FOUND)
    set(conduit_relay_fortran_modules
        ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit_relay.mod)

    # Setup install to copy the fortran modules 
    install(FILES 
            ${conduit_relay_fortran_modules}
            DESTINATION include/conduit)
endif()


################################################################
# If we have mpi, add the conduit relay mpi library
################################################################

if(MPI_FOUND)

#
# Specify relay mpi headers
#
set(conduit_relay_mpi_headers
    conduit_relay_mpi.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_config.h)

#
# Specify relay mpi sources
#
set(conduit_relay_mpi_sources conduit_relay_mpi.cpp)

#
# Specify the relay mpi deps
#
set(conduit_relay_mpi_deps conduit mpi)

#
# setup conduit relay mpi lib
#
add_compiled_library(NAME conduit_relay_mpi
                     EXPORT conduit
                     HEADERS ${conduit_relay_mpi_headers}
                     SOURCES ${conduit_relay_mpi_sources}
                     DEPENDS_ON ${conduit_relay_mpi_deps}
                     HEADERS_DEST_DIR include/conduit)


################################################################
# If we have mpi, add the conduit relay mpi io library
#
# We put the io functions into a separate library so we do not
# disturb applications that may already be linking with
# relay+relay_mpi.
################################################################

#
# Specify relay mpi io sources
#
set(conduit_relay_mpi_io_sources conduit_relay_mpi_io.cpp
                                 conduit_relay_io_identify_protocol.cpp)

#
# Specify relay mpi io headers
#
set(conduit_relay_mpi_io_headers conduit_relay_mpi_io.hpp)

#
# Specify the relay mpi io deps
#
set(conduit_relay_mpi_io_deps conduit mpi)

#
# Specify relay mpi c headers
#
set(conduit_relay_mpi_io_c_headers
    c/conduit_relay_mpi_io.h)
#
# Specify relay mpi c sources
#
set(conduit_relay_mpi_io_c_sources
    c/conduit_relay_mpi_io_c.cpp)

if(SILO_FOUND)
    list(APPEND conduit_relay_mpi_io_headers conduit_relay_mpi_io_silo.hpp)
    list(APPEND conduit_relay_mpi_io_sources conduit_relay_io_silo.cpp)
    list(APPEND conduit_relay_mpi_io_deps silo)
endif()
if(HDF5_FOUND)
    list(APPEND conduit_relay_mpi_io_headers conduit_relay_mpi_io_hdf5.hpp)
    list(APPEND conduit_relay_mpi_io_sources conduit_relay_io_hdf5.cpp)
    list(APPEND conduit_relay_mpi_io_deps hdf5)
endif()
if(ADIOS_FOUND)
    list(APPEND conduit_relay_mpi_io_headers conduit_relay_mpi_io_adios.hpp)
    list(APPEND conduit_relay_mpi_io_sources conduit_relay_io_adios.cpp)
    # Link with parallel versions of ADIOS
    list(APPEND conduit_relay_mpi_io_deps adios_mpi)
endif()

#
# setup conduit relay mpi io lib
#
add_compiled_library(NAME conduit_relay_mpi_io
                     EXPORT conduit
                     HEADERS ${conduit_relay_mpi_io_headers} ${conduit_relay_mpi_io_c_headers}
                     SOURCES ${conduit_relay_mpi_io_sources} ${conduit_relay_mpi_io_c_sources}
                     DEPENDS_ON ${conduit_relay_mpi_io_deps}
                     HEADERS_DEST_DIR include/conduit)

blt_add_target_compile_flags(TO conduit_relay_mpi_io FLAGS "-DCONDUIT_RELAY_IO_MPI_ENABLED")


endif() # end if MPI_FOUND
