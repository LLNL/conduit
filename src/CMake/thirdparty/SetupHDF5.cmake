###############################################################################
# Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
#
# Setup HDF5
#

# first Check for HDF5_DIR

if(NOT HDF5_DIR)
    MESSAGE(FATAL_ERROR "HDF5 support needs explicit HDF5_DIR")
endif()

# CMake's FindHDF5 module uses the HDF5_ROOT env var
set(HDF5_ROOT ${HDF5_DIR})

if(NOT WIN32)
    set(ENV{HDF5_ROOT} ${HDF5_ROOT}/bin)
    # Use CMake's FindHDF5 module, which uses hdf5's compiler wrappers to extract
    # all the info about the hdf5 install
    include(FindHDF5)

else()
    # CMake's FindHDF5 module is buggy on windows and will put the dll
    # in HDF5_LIBRARY.  Instead, use the 'CONFIG' signature of find_package
    # with appropriate hints for where cmake can find hdf5-config.cmake.
    find_package(HDF5 CONFIG 
                 REQUIRED
                 HINTS ${HDF5_DIR}/cmake/hdf5 
                       ${HDF5_DIR}/lib/cmake/hdf5
                       ${HDF5_DIR}/share/cmake/hdf5)
endif()

# FindHDF5/find_package sets HDF5_DIR to it's installed CMake info if it exists
# we want to keep HDF5_DIR as the root dir of the install to be 
# consistent with other packages

set(HDF5_DIR ${HDF5_ROOT} CACHE PATH "" FORCE)

#
# Sanity check to alert us if some how we found an hdf5 instance
# in an unexpected location.  
#
message(STATUS "Checking that found HDF5_INCLUDE_DIRS are in HDF5_DIR")

foreach(IDIR ${HDF5_INCLUDE_DIRS})
    if("${IDIR}" MATCHES "${HDF5_DIR}")
        message(STATUS " ${IDIR} includes HDF5_DIR")
        
    else()
        message(FATAL_ERROR " ${IDIR} does not include HDF5_DIR")
    endif()
endforeach()

#
# filter HDF5_LIBRARIES to remove hdf5_hl if it exists
# we don't use hdf5_hl, but if we link with it will become
# a transitive dependency
#
set(HDF5_HL_LIB FALSE)
foreach(LIB ${HDF5_LIBRARIES})
    if("${LIB}" MATCHES "hdf5_hl")
        set(HDF5_HL_LIB ${LIB})
    endif()
endforeach()

if(HDF5_HL_LIB)
    message(STATUS "Removing hdf5_hl from HDF5_LIBRARIES")
    list(REMOVE_ITEM HDF5_LIBRARIES ${HDF5_HL_LIB})
endif()


#
# Display main hdf5 cmake vars
#
message(STATUS "HDF5 Include Dirs ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 Libraries    ${HDF5_LIBRARIES}")





