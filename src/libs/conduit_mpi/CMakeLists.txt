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
# For details, see: http://cyrush.github.io/conduit.
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
# Add the conduit-mpi library
################################

ENABLE_WARNINGS()

#
# Specify all headers
#            
set(conduit_mpi_headers 
    conduit_mpi.hpp
    Conduit_MPI_Exports.hpp)

#
# Specify all sources
#
set(conduit_mpi_sources conduit_mpi.cpp)

include_directories(${MPI_CXX_INCLUDE_PATH})

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
add_library(conduit_mpi SHARED ${conduit_mpi_sources} ${conduit_mpi_headers})

# guard against empty mpi params
if(NOT "${MPI_CXX_COMPILE_FLAGS}" STREQUAL "")
    set_target_properties(conduit_mpi 
                                PROPERTIES COMPILE_FLAGS            
                                ${MPI_CXX_COMPILE_FLAGS} )
endif()

if(NOT "${MPI_CXX_LINK_FLAGS}" STREQUAL "")
    set_target_properties(conduit_mpi 
                                PROPERTIES LINK_FLAGS
                                ${MPI_CXX_LINK_FLAGS} )
endif()

#
# Link with conduit and silo (if enabled)
#
target_link_libraries(conduit_mpi conduit ${MPI_CXX_LIBRARIES})


##################################
# Install Targets for conduit lib
##################################

install(FILES ${conduit_mpi_headers} DESTINATION include)
install(TARGETS conduit_mpi
  EXPORT conduit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)

