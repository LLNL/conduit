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
# Standard CMake Options
################################


# Fail if someone tries to config an in-source build.
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
   message(FATAL_ERROR "In-source builds are not supported. Please remove CMakeCache.txt from the 'src' dir and configure an out-of-source build in another directory.")
endif()

# enable creation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# always use position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

macro(ENABLE_WARNINGS)
    # set the warning levels we want to abide by
    if(CMAKE_BUILD_TOOL MATCHES "(msdev|devenv|nmake)")
        add_definitions(/W2)
    else()
        if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR
            "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR 
            "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            # use these flags for clang, gcc, or icc
            add_definitions(-Wall -Wextra)
        endif()
    endif()
endmacro()


################################
# Shared vs Static Libs
################################
if(BUILD_SHARED_LIBS)
    message(STATUS "Building shared libraries (BUILD_SHARED_LIBS == ON)")
else()
    message(STATUS "Building static libraries (BUILD_SHARED_LIBS == OFF)")
endif()

################################
# Coverage Flags
################################
if(ENABLE_COVERAGE)
    message(STATUS "Building using coverage flags (ENABLE_COVERAGE == ON)")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} --coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
else()
    message(STATUS "Building without coverage flags (ENABLE_COVERAGE == OFF)")
endif()

################################
# RPath Settings
################################

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if("${isSystemDir}" STREQUAL "-1")
   set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
endif()

################################
# Standard CTest Options
################################
if(ENABLE_TESTS)
    include(CTest)
endif()

######################################################################################
# Provide macros to simplify adding compile and link flags to a target
######################################################################################

macro(add_target_compile_flags)
    set(options)
    set(singleValuedArgs TARGET FLAGS)
    set(multiValuedArgs)

    ## parse the arguments to the macro
    cmake_parse_arguments(args
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )
    
    if(NOT "${args_FLAGS}" STREQUAL "")
        # get prev flags
        get_target_property(_COMP_FLAGS ${args_TARGET} COMPILE_FLAGS)
        if(NOT _COMP_FLAGS)
            set(_COMP_FLAGS "")
        endif()
        # append new flags
        set(_COMP_FLAGS "${args_FLAGS} ${_COMP_FLAGS}")
        set_target_properties(${args_TARGET} PROPERTIES COMPILE_FLAGS "${_COMP_FLAGS}" )
    endif()
    
endmacro()

macro(add_target_link_flags)
    set(options)
    set(singleValuedArgs TARGET FLAGS)
    set(multiValuedArgs)

    ## parse the arguments to the macro
    cmake_parse_arguments(args
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )
    
    if(NOT "${args_FLAGS}" STREQUAL "")
        # get prev flags
        get_target_property(_LINK_FLAGS ${args_TARGET} LINK_FLAGS)
        if(NOT _LINK_FLAGS)
            set(_LINK_FLAGS "")
        endif()
        # append new flag
        set(_LINK_FLAGS "${args_FLAGS} ${_LINK_FLAGS}")
        set_target_properties(${args_TARGET} PROPERTIES LINK_FLAGS "${_LINK_FLAGS}" )
    endif()
    
endmacro()


