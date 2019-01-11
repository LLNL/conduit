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
# Standard CMake Options
################################


# Fail if someone tries to config an in-source build.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   message(FATAL_ERROR "In-source builds are not supported. Please remove "
                       "CMakeCache.txt from the 'src' dir and configure an "
                       "out-of-source build in another directory.")
endif()

# enable creation of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# always use position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CMake build tool name: ${CMAKE_BUILD_TOOL}")

macro(ENABLE_WARNINGS)
    # set the warning levels we want to abide by
    if("${CMAKE_BUILD_TOOL}" MATCHES "(msdev|devenv|nmake|MSBuild)")
        add_definitions(/W4)
    else()
        if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
            "${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU"   OR
            "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
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

if(WIN32 AND BUILD_SHARED_LIBS)
    set(CONDUIT_WINDOWS_DLL_EXPORTS TRUE)
endif()

if( BLT_CXX_STD STREQUAL "c++98" )
    message(STATUS "C++11 support disabled")
else()
    set(CONDUIT_USE_CXX11 1)
    message(STATUS "C++11 support enabled (CONDUIT_USE_CXX11 == 1)")
endif()

#######################################
# Global Helpers (clear every config)
#######################################
set(CONDUIT_INSTALL_PREFIX CACHE STRING "" FORCE)
set(CONDUIT_MAKE_EXTRA_LIBS CACHE STRING "" FORCE)

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
# Standard CTest Options
################################
if(ENABLE_TESTS)
    set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full --gen-suppressions=all")
    file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/cmake/valgrind.supp" MEMORYCHECK_SUPPRESSIONS_FILE)
    include(CTest)
    message(STATUS "Memcheck suppressions file: ${MEMORYCHECK_SUPPRESSIONS_FILE}")
endif()

##############################################################################
# Try to extract the current git sha
#
# This solution is derived from:
#  http://stackoverflow.com/a/21028226/203071
#
# This does not have full dependency tracking - it wont auto update when the
# git HEAD changes or when a branch is checked out, unless a change causes
# cmake to reconfigure.
#
# However, this limited approach will still be useful in many cases, 
# including building and for installing  conduit as a tpl
#
##############################################################################
find_package(Git)
if(GIT_FOUND)
  message("git executable: ${GIT_EXECUTABLE}")
  execute_process(COMMAND
    "${GIT_EXECUTABLE}" describe --match=NeVeRmAtCh --always --abbrev=40 --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE CONDUIT_GIT_SHA1
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "Repo SHA1:" ${CONDUIT_GIT_SHA1})
endif()


###############################################################################
# Provide macros to simplify creating libs
###############################################################################
macro(add_compiled_library)
    set(options OBJECT)
    set(singleValuedArgs NAME EXPORT HEADERS_DEST_DIR LIB_DEST_DIR )
    set(multiValuedArgs  HEADERS SOURCES DEPENDS_ON)

    ## parse the arguments to the macro
    cmake_parse_arguments(args
         "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN} )

    #############################
    # add lib target
    #############################

    #
    # Note: headers are added here so they show up in project file generators
    # (such as xcode, eclipse, etc)
    #

    # if OBJECT is present, create an object library instead of a
    # standard library.

    if(${args_OBJECT})
        add_library(${args_NAME} OBJECT
                    ${args_SOURCES}
                    ${args_HEADERS})

    else()
        
        # general case build libs into lib dir
        set(lib_output_dir ${CMAKE_BINARY_DIR}/lib)

        # on windows we need the libs to live next to the tests
        if(WIN32)
            set(lib_output_dir ${CMAKE_BINARY_DIR}/bin)
        endif()
        
        blt_add_library( NAME        ${args_NAME}
                         SOURCES     ${args_SOURCES} 
                         HEADERS     ${args_HEADERS}
                         OUTPUT_DIR  ${lib_output_dir}
                         DEPENDS_ON  ${args_DEPENDS_ON})
    endif()

    #############################
    # install and export setup
    #############################
    # if export is set, install our lib

    # object libraries don't need to be installed
    if(NOT ${args_OBJECT})
        # if we have headers, install them
        if(NOT "${args_HEADERS}" STREQUAL "")
            if(NOT "${args_HEADERS_DEST_DIR}" STREQUAL "")
                install(FILES ${args_HEADERS}
                        DESTINATION ${args_HEADERS_DEST_DIR})
            else()
                install(FILES ${args_HEADERS}
                        DESTINATION include)
            endif()
        endif()
        # install our lib
        if(NOT "${args_LIB_DEST}" STREQUAL "")
            install(TARGETS ${args_NAME}
                    EXPORT ${args_EXPORT}
                    LIBRARY DESTINATION ${args_LIB_DEST_DIR}
                    ARCHIVE DESTINATION ${args_LIB_DEST_DIR}
                    RUNTIME DESTINATION ${args_LIB_DEST_DIR})
        else()
            install(TARGETS ${args_NAME}
                    EXPORT ${args_EXPORT}
                    LIBRARY DESTINATION lib
                    ARCHIVE DESTINATION lib
                    RUNTIME DESTINATION lib)
        endif()
    endif()

endmacro()


###############################################################################
# Provide macros to simplify adding compile and link flags to a target
###############################################################################

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
        set_target_properties(${args_TARGET}
                              PROPERTIES COMPILE_FLAGS "${_COMP_FLAGS}" )
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
        set_target_properties(${args_TARGET}
                              PROPERTIES LINK_FLAGS "${_LINK_FLAGS}" )
    endif()

endmacro()

###############################################################################
# This macro converts a cmake path to a platform specific string literal
# usable in C++. (For example, on windows C:/Path will be come C:\\Path)
###############################################################################

macro(convert_to_native_escaped_file_path path output)
    file(TO_NATIVE_PATH ${path} ${output})
    string(REPLACE "\\" "\\\\"  ${output} "${${output}}")
endmacro()


