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
# Add the conduit-io library
################################

ENABLE_WARNINGS()

if(HDF5_FOUND)
  SET(CONDUIT_IO_ENABLE_HDF5 TRUE)
endif()

if(SILO_FOUND)
  SET(CONDUIT_IO_ENABLE_SILO TRUE)
endif()

set(CONDUIT_WEB_CLIENT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/web_clients)

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/Conduit_IO_Config.hpp.in"
                "${CMAKE_CURRENT_BINARY_DIR}/Conduit_IO_Config.hpp")

#
# Specify all headers
#            
set(conduit_io_headers 
    conduit_io.hpp
    conduit_web.hpp
    Conduit_IO_Exports.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/Conduit_IO_Config.hpp)

#
# Specify all sources
#
set(conduit_io_sources 
    conduit_io.cpp
    conduit_web.cpp)
    
if(HDF5_FOUND)
    list(APPEND conduit_io_headers conduit_hdf5.hpp)
    list(APPEND conduit_io_sources conduit_hdf5.cpp)
endif()
    
if(SILO_FOUND)
    list(APPEND conduit_io_headers conduit_silo.hpp)
    list(APPEND conduit_io_sources conduit_silo.cpp)
endif()

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
add_library(conduit_io SHARED ${conduit_io_sources} ${conduit_io_headers})

#
# Link with conduit and silo (if enabled)
#
target_link_libraries(conduit_io conduit civetweb)

if(SILO_FOUND)
    target_link_libraries(conduit_io conduit ${SILO_LIBRARIES})
endif()

if(HDF5_FOUND)
    target_link_libraries(conduit_io conduit ${HDF5_LIBRARIES})
endif()


##################################
# Install Targets for conduit lib
##################################

install(FILES ${conduit_io_headers} DESTINATION include)
install(TARGETS conduit_io
  EXPORT conduit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)

