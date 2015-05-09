###############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC.
# 
# Produced at the Lawrence Livermore National Laboratory
# 
# LLNL-CODE-666778
# 
# All rights reserved.
# 
# This file is part of Conduit. 
# 
# For details, see https://lc.llnl.gov/conduit/.
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

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/Conduit_Config.hpp.in"
                "${CMAKE_CURRENT_BINARY_DIR}/Conduit_Config.hpp")
#
# Specify all headers
#            
set(conduit_headers
    Bitwidth_Style_Types.hpp
    DataArray.hpp
    Endianness.hpp
    Node.hpp
    conduit.hpp
    Core.hpp
    DataType.hpp
    Generator.hpp
    Error.hpp
    NodeIterator.hpp
    Schema.hpp
    Utils.hpp
    Conduit_Exports.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/Conduit_Config.hpp
    )

#
# Specify all sources
#
set(conduit_sources
    Core.cpp
    Error.cpp
    Endianness.cpp 
    DataType.cpp
    DataArray.cpp
    Generator.cpp
    Node.cpp
    NodeIterator.cpp
    Schema.cpp
    Utils.cpp
    )

#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
add_library(conduit SHARED ${conduit_sources} ${conduit_headers})

################################
# Add python wrappers if python
# support was selected
################################
if(ENABLE_PYTHON)
    add_subdirectory(python)
endif()

##################################
# Install Targets for conduit lib
##################################

install(FILES ${conduit_headers} DESTINATION include)
install(TARGETS conduit
  EXPORT conduit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)

