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
# Add the blueprint library
################################

ENABLE_WARNINGS()

#
# Specify all headers
#            
set(blueprint_headers 
    blueprint.hpp
    blueprint_mesh.hpp
    Blueprint_Exports.hpp)

#
# Specify all sources
#
set(blueprint_sources 
    blueprint.cpp
    blueprint_mesh.cpp)

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
if(BUILD_SHARED_LIBS)
    add_library(blueprint SHARED ${blueprint_sources} ${blueprint_headers})
else()
    add_library(blueprint STATIC ${blueprint_sources} ${blueprint_headers})
endif()

#
# Link with conduit
#
target_link_libraries(blueprint conduit)

##################################
# Install Targets for conduit 
##################################

install(FILES ${blueprint_headers} DESTINATION include)
install(TARGETS blueprint
  EXPORT conduit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)



