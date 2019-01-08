###############################################################################
# Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
# Setup ADIOS
#

# first Check for ADIOS_DIR

if(NOT ADIOS_DIR)
    MESSAGE(FATAL_ERROR "ADIOS support needs explicit ADIOS_DIR")
endif()

MESSAGE(STATUS "Looking for ADIOS using ADIOS_DIR = ${ADIOS_DIR}")

# CMake's FindADIOS module uses the ADIOS_ROOT env var
set(ADIOS_ROOT ${ADIOS_DIR})
set(ENV{ADIOS_ROOT} ${ADIOS_ROOT})

FUNCTION(FIND_ADIOS COMPS FOUNDVAR INCVAR LIBVAR)
    # Set the components that we want.
    SET(ADIOS_FIND_COMPONENTS ${COMPS})

    # Use CMake's FindADIOS module, which uses ADIOS's compiler wrappers to extract
    # all the info about the ADIOS install
    include(${ADIOS_DIR}/etc/FindADIOS.cmake)

    # FindADIOS sets ADIOS_DIR to it's installed CMake info if it exists
    # we want to keep ADIOS_DIR as the root dir of the install to be 
    # consistent with other packages

    set(ADIOS_DIR ${ADIOS_ROOT} CACHE PATH "" FORCE)
    # not sure why we need to set this, but we do
    #set(ADIOS_FOUND TRUE CACHE PATH "" FORCE)

    if(NOT ADIOS_FOUND)
        message(FATAL_ERROR "ADIOS_DIR is not a path to a valid ADIOS install")
    endif()

    # Set the return variables.
    SET(${FOUNDVAR} ${ADIOS_FOUND} PARENT_SCOPE)
    SET(${INCVAR} ${ADIOS_INCLUDE_DIRS} PARENT_SCOPE)
    SET(${LIBVAR} ${ADIOS_LIBRARIES} PARENT_SCOPE)

    # Unset some variables.
    unset(ADIOS_INCLUDE_DIRS)
    unset(ADIOS_LIBRARIES)
    unset(ADIOS_FOUND)
ENDFUNCTION(FIND_ADIOS)

IF(ENABLE_MPI)
    # Find the MPI-enabled ADIOS library variants that we want.
    FIND_ADIOS("" ADIOS_FOUND ADIOS_INC ADIOS_LIB)
    FIND_ADIOS("readonly" ADIOSREAD_FOUND ADIOSREAD_INC ADIOSREAD_LIB)

    set(adios_mpi_includes ${ADIOS_INC}  ${ADIOSREAD_INC})
    set(adios_mpi_libs ${ADIOS_LIB} ${ADIOSREAD_LIB})

    # on linux we need to link threading libs as well
    if(UNIX AND NOT APPLE)
        list(APPEND adios_mpi_libs rt ${CMAKE_THREAD_LIBS_INIT})
    endif()

    # bundle both std lib  and read only libs as 'adios_mpi'
    blt_register_library(NAME adios_mpi
                         INCLUDES ${adios_mpi_includes}
                         LIBRARIES ${adios_mpi_libs})

     # generate libs and include strs to export to config.mk
     set(adios_mpi_make_incs "")
     set(adios_mpi_make_libs "")
     
     foreach(inc_val ${adios_mpi_includes})
         set(adios_mpi_make_incs "${adios_mpi_make_incs} -I${inc_val}")
     endforeach()

     foreach(lib_val ${adios_mpi_libs})
         set(adios_mpi_make_libs "${adios_mpi_make_libs} ${lib_val}")
     endforeach()

     set("CONDUIT_ADIOS_MPI_MAKE_INCLUDES_STR" "${adios_mpi_make_incs}")
     set("CONDUIT_ADIOS_MPI_MAKE_LIBS_STR" "${adios_mpi_make_libs}")

 ENDIF(ENABLE_MPI)

# Find the serial ADIOS library variants that we want.
FIND_ADIOS("sequential" ADIOS_FOUND ADIOS_SEQ_INC ADIOS_SEQ_LIB)
FIND_ADIOS("sequential;readonly" ADIOSREAD_FOUND ADIOSREAD_SEQ_INC ADIOSREAD_SEQ_LIB)

set(adios_nompi_includes ${ADIOS_SEQ_INC} ${ADIOSREAD_SEQ_INC})
set(adios_nompi_libs  ${ADIOS_SEQ_LIB} ${ADIOSREAD_SEQ_LIB})

# on linux we need to link threading libs as well
if(UNIX AND NOT APPLE)
    list(APPEND adios_nompi_libs rt ${CMAKE_THREAD_LIBS_INIT})
endif()

# generate libs and include strs to export to config.mk
set(adios_nompi_make_incs "")
set(adios_nompi_make_libs "")

foreach(inc_val ${adios_nompi_includes})
    set(adios_nompi_make_incs "${adios_nompi_make_incs} -I${inc_val}")
endforeach()

foreach(lib_val ${adios_mpi_libs})
    set(adios_nompi_make_libs "${adios_nompi_make_libs} ${lib_val}")
endforeach()

set("CONDUIT_ADIOS_NOMPI_MAKE_INCLUDES_STR" "${adios_nompi_make_incs}")
set("CONDUIT_ADIOS_NOMPI_MAKE_LIBS_STR" "${adios_nompi_make_libs}")

# bundle both seq and seq read only libs as 'adios_nompi'
blt_register_library(NAME adios_nompi
                     INCLUDES  ${adios_nompi_includes}
                     LIBRARIES ${adios_nompi_libs})


# Print out some results.
MESSAGE(STATUS "  ADIOS_INC=${ADIOS_INC}")
MESSAGE(STATUS "  ADIOS_LIB=${ADIOS_LIB}")

MESSAGE(STATUS "  ADIOSREAD_INC=${ADIOSREAD_INC}")
MESSAGE(STATUS "  ADIOSREAD_LIB=${ADIOSREAD_LIB}")

MESSAGE(STATUS "  ADIOS_SEQ_INC=${ADIOS_SEQ_INC}")
MESSAGE(STATUS "  ADIOS_SEQ_LIB=${ADIOS_SEQ_LIB}")

MESSAGE(STATUS "  ADIOSREAD_SEQ_INC=${ADIOSREAD_SEQ_INC}")
MESSAGE(STATUS "  ADIOSREAD_SEQ_LIB=${ADIOSREAD_SEQ_LIB}")

