###############################################################################
# Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see: http://llnl.github.io/conduit/.
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

set(CONDUIT_RELAY_WEB_CLIENT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/web_clients)

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/relay_config.hpp.in"
                "${CMAKE_CURRENT_BINARY_DIR}/relay_config.hpp")

#
# Specify all headers
#            
set(conduit_relay_headers 
    relay.hpp
    relay_io.hpp
    relay_web.hpp
    relay_web_visualizer.hpp
    relay_exports.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/relay_config.hpp)

#
# Specify all sources
#
set(conduit_relay_sources 
    relay.cpp
    relay_io.cpp
    relay_web.cpp
    relay_web_visualizer.cpp)
    
if(HDF5_FOUND)
    list(APPEND conduit_relay_headers relay_hdf5.hpp)
    list(APPEND conduit_relay_sources relay_hdf5.cpp)
endif()
    
if(SILO_FOUND)
    list(APPEND conduit_relay_headers relay_silo.hpp)
    list(APPEND conduit_relay_sources relay_silo.cpp)
endif()

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
if(BUILD_SHARED_LIBS)
    add_library(conduit_relay SHARED ${conduit_relay_sources} ${conduit_relay_headers})
else()
    add_library(conduit_relay STATIC ${conduit_relay_sources} ${conduit_relay_headers})
endif()

#
# Link with conduit 
#
target_link_libraries(conduit_relay conduit civetweb)


#
# Link with silo and hdf5 if these are enabled
#
if(SILO_FOUND)
    target_link_libraries(conduit_relay conduit ${SILO_LIBRARIES})
endif()

if(HDF5_FOUND)
    target_link_libraries(conduit_relay conduit ${HDF5_LIBRARIES})
endif()

#########################################
# Install Targets for conduit relay
#########################################

install(FILES ${conduit_relay_headers} DESTINATION include)
install(TARGETS conduit_relay
  EXPORT conduit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)


################################################################
# If we have mpi, add the conduit relay mpi library
################################################################

if(MPI_FOUND)

#
# Specify relay mpi headers
#            
set(conduit_relay_mpi_headers 
    relay_mpi.hpp
    relay_exports.hpp)

#
# Specify relay mpi sources
#
set(conduit_relay_mpi_sources relay_mpi.cpp)

include_directories(${MPI_CXX_INCLUDE_PATH})

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
if(BUILD_SHARED_LIBS)
    add_library(conduit_relay_mpi SHARED ${conduit_relay_mpi_sources} ${conduit_relay_mpi_headers})
else()
    add_library(conduit_relay_mpi STATIC ${conduit_relay_mpi_sources} ${conduit_relay_mpi_headers})
endif()
        

# guard against empty mpi params
if(NOT "${MPI_CXX_COMPILE_FLAGS}" STREQUAL "")
    set_target_properties(conduit_relay_mpi 
                                PROPERTIES COMPILE_FLAGS
                                ${MPI_CXX_COMPILE_FLAGS} )
endif()

if(NOT "${MPI_CXX_LINK_FLAGS}" STREQUAL "")
    set_target_properties(conduit_relay_mpi 
                                PROPERTIES LINK_FLAGS
                                ${MPI_CXX_LINK_FLAGS} )
endif()

#
# Link with conduit and MPI
#
target_link_libraries(conduit_relay_mpi conduit ${MPI_CXX_LIBRARIES})


#########################################
# Install Targets for conduit relay mpi
#########################################

install(FILES ${conduit_relay_mpi_headers} DESTINATION include)
install(TARGETS conduit_relay_mpi
  EXPORT conduit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)

endif() # end if MPI_FOUND


