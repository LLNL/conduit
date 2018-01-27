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


################################
# Guards for Fortran support.
################################
if(ENABLE_FORTRAN)
    set(CMAKE_Fortran_FORMAT "FREE")
    if(CMAKE_Fortran_COMPILER)
        MESSAGE(STATUS  "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
        set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/fortran)
        
        # make sure the fortran compiler can see the module files it
        # generates
        include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
        try_compile(Fortran_COMPILER_SUPPORTS_CLASS ${CMAKE_BINARY_DIR}
                    ${CMAKE_SOURCE_DIR}/cmake/tests/fortran_test_obj_support.f
                    CMAKE_FLAGS "-DCMAKE_Fortran_FORMAT=FREE"
                    OUTPUT_VARIABLE OUTPUT)

        set(ENABLE_FORTRAN_OBJ_INTERFACE ${Fortran_COMPILER_SUPPORTS_CLASS})
        
    elseif(CMAKE_GENERATOR STREQUAL Xcode)
        MESSAGE(STATUS "Disabling Fortran support: ENABLE_FORTRAN is true, "
                       "but the Xcode CMake Generator does not support Fortran.")
        set(ENABLE_FORTRAN OFF)
    else()
        MESSAGE(FATAL_ERROR "ENABLE_FORTRAN is true, but a Fortran compiler wasn't found.")
    endif()
    # a this point, if ENABLE_FORTRAN is still on, we have found a Fortran compiler
    if(ENABLE_FORTRAN)
        set(FORTRAN_FOUND 1)
    endif()
endif()


