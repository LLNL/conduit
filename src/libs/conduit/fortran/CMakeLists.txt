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
# For details, see: http://scalability-llnl.github.io/conduit/.
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

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/Bitwidth_Style_Types.inc.in"
                "${CMAKE_CURRENT_BINARY_DIR}/Bitwidth_Style_Types.inc")

# Allow fortran compiler to include the Bitwidth_Style_Types.inc file
# Note: Unlike .h files, the .inc file will not be needed by client code
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#
# Specify fortran sources
#
set(conduit_fortran_sources
    conduit_fortran.F)
    
if(ENABLE_FORTRAN_OBJ_INTERFACE)
    list(APPEND conduit_fortran_sources conduit_fortran_obj.f)
endif()
#
# Note: headers are added here so they show up in project file generators
# (such as xcode, eclipse, etc)
#
add_library(conduit_fortran SHARED 
            ${conduit_fortran_sources}
            )

target_link_libraries(conduit_fortran  conduit)

set_target_properties(conduit_fortran PROPERTIES Fortran_FORMAT "FREE")

##################################
# Install Targets for conduit lib
##################################

set(conduit_fortran_modules
    ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit.mod)

if(ENABLE_FORTRAN_OBJ_INTERFACE)
    list(APPEND conduit_fortran_modules ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit_obj.mod)
endif()

# Note: We don't need to install the .inc file
install(FILES 
        ${conduit_fortran_modules}
        DESTINATION include)


install(TARGETS conduit_fortran
  EXPORT conduit
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib
)


