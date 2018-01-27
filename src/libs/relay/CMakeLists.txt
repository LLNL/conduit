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
    conduit_relay_web.hpp
    conduit_relay_web_node_viewer_server.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_exports.h
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_config.h)

#
# Specify c headers
#
set(conduit_relay_c_headers
    c/conduit_relay.h)

#
# Specify cpp sources
#
set(conduit_relay_sources
    conduit_relay.cpp
    conduit_relay_io.cpp
    conduit_relay_web.cpp
    conduit_relay_web_node_viewer_server.cpp)

#
# Specify c sources
#
set(conduit_relay_c_sources
    c/conduit_relay_c.cpp)


if(HDF5_FOUND)
    list(APPEND conduit_relay_headers conduit_relay_hdf5.hpp)
    list(APPEND conduit_relay_sources conduit_relay_hdf5.cpp)
endif()

if(SILO_FOUND)
    list(APPEND conduit_relay_headers conduit_relay_silo.hpp)
    list(APPEND conduit_relay_sources conduit_relay_silo.cpp)
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
# target_link_libraries(conduit_relay PUBLIC conduit)


set(conduit_relay_deps conduit)

if(UNIX AND NOT APPLE)
    # we need these for civetweb on linux, we may need similar libs
    # on windows (OSX appears ok without them)
    #target_link_libraries(conduit_relay PUBLIC dl rt ${CMAKE_THREAD_LIBS_INIT})
    list(APPEND conduit_relay_deps dl rt ${CMAKE_THREAD_LIBS_INIT})
endif()

#
# Link with silo and hdf5 if these are enabled
#
if(SILO_FOUND)
    list(APPEND conduit_relay_deps silo)
    #target_link_libraries(conduit_relay PUBLIC ${SILO_LIBRARIES})
endif()

if(HDF5_FOUND)
    list(APPEND conduit_relay_deps hdf5)
    #target_link_libraries(conduit_relay PUBLIC ${HDF5_LIBRARIES})
endif()




add_compiled_library(NAME conduit_relay
                     EXPORT conduit
                     HEADERS ${conduit_relay_headers} ${conduit_relay_c_headers}
                     SOURCES ${conduit_relay_sources} ${conduit_relay_c_sources} ${conduit_relay_fortran_sources}
                             $<TARGET_OBJECTS:conduit_civetweb>
                     DEPENDS_ON ${conduit_relay_deps}
                     HEADERS_DEST_DIR include/conduit)

#  todo?
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

#add_executable(conduit_relay_node_viewer conduit_relay_web_node_viewer_exe.cpp)
#target_link_libraries(conduit_relay_node_viewer conduit_relay)

# add install target
install(TARGETS conduit_relay_node_viewer
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

# include_directories(${MPI_CXX_INCLUDE_PATH})

#
# setup conduit relay mpi lib
#
add_compiled_library(NAME conduit_relay_mpi
                     EXPORT conduit
                     HEADERS ${conduit_relay_mpi_headers}
                     SOURCES ${conduit_relay_mpi_sources}
                     DEPENDS_ON conduit mpi
                     HEADERS_DEST_DIR include/conduit)


#
# add compile and link flags necessary to use mpi
# #
# add_target_compile_flags(TARGET conduit_relay_mpi
#                          FLAGS ${MPI_CXX_COMPILE_FLAGS})
#
# add_target_link_flags(TARGET conduit_relay_mpi
#                       FLAGS ${MPI_CXX_LINK_FLAGS})
#
#
# #
# # Link with conduit and MPI
# #
# target_link_libraries(conduit_relay_mpi conduit ${MPI_CXX_LIBRARIES})


endif() # end if MPI_FOUND
