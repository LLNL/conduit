###############################################################################
# Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

# find the absolute path w/ symlinks resolved of the passed HDF5_DIR, 
# since sanity checks later need to compare against the real path
get_filename_component(HDF5_DIR_REAL "${HDF5_DIR}" REALPATH)
message(STATUS "Looking for HDF5 at: " ${HDF5_DIR_REAL})

# CMake's FindHDF5 module uses the HDF5_ROOT env var
set(HDF5_ROOT ${HDF5_DIR_REAL})

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
                       ${HDF5_DIR}/share/cmake/hdf5
                       ${HDF5_DIR}/cmake)
endif()

# FindHDF5/find_package sets HDF5_DIR to it's installed CMake info if it exists
# we want to keep HDF5_DIR as the root dir of the install to be 
# consistent with other packages

# find the absolute path w/ symlinks resolved of the passed HDF5_DIR, 
# since sanity checks later need to compare against the real path
get_filename_component(HDF5_DIR_REAL "${HDF5_ROOT}" REALPATH)

set(HDF5_DIR ${HDF5_DIR_REAL} CACHE PATH "" FORCE)
message(STATUS "HDF5_DIR_REAL=${HDF5_DIR_REAL}")
#
# Sanity check to alert us if some how we found an hdf5 instance
# in an unexpected location.  
#
message(STATUS "Checking that found HDF5_INCLUDE_DIRS are in HDF5_DIR")

#
# HDF5_INCLUDE_DIRS may also include paths to external lib headers 
# (such as szip), so we check that *at least one* of the includes
# listed in HDF5_INCLUDE_DIRS exists in the HDF5_DIR specified. 
#

# HDF5_INCLUDE_DIR is deprecated, but there are still some cases
# where HDF5_INCLUDE_DIR is set, but HDF5_INCLUDE_DIRS is not
if(NOT HDF5_INCLUDE_DIRS)
    if(HDF5_INCLUDE_DIR)
        set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
    else()
        message(FATAL_ERROR "FindHDF5 did not provide HDF5_INCLUDE_DIRS or HDF5_INCLUDE_DIR.")
    endif()
endif()

message(STATUS "HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS}")
set(check_hdf5_inc_dir_ok 0)
foreach(IDIR ${HDF5_INCLUDE_DIRS})
    
    # get real path of the include dir 
    # w/ abs and symlinks resolved
    get_filename_component(IDIR_REAL "${IDIR}" REALPATH)
    # check if idir_real is a substring of hdf5_dir

    if("${IDIR_REAL}" MATCHES "${HDF5_DIR}")
        message(STATUS " ${IDIR_REAL} includes HDF5_DIR (${HDF5_DIR})")
        set(check_hdf5_inc_dir_ok 1)
    endif()
endforeach()

if(NOT check_hdf5_inc_dir_ok)
    message(FATAL_ERROR " ${HDF5_INCLUDE_DIRS} does not include HDF5_DIR")
endif()

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
message(STATUS "HDF5 is parallel  ${HDF5_IS_PARALLEL}")


blt_register_library(NAME hdf5
                     INCLUDES ${HDF5_INCLUDE_DIRS}
                     LIBRARIES ${HDF5_LIBRARIES} )



