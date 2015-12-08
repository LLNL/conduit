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
# Conduit Unit Tests
################################

################################
# Core (conduit lib) Unit Tests
################################
set(TPL_TESTS gtest_smoke
              rapidjson_smoke
              libb64_smoke)


################################
# Optional Unit Tests
################################
set(PYTHON_GPREF_TOOLS_TESTS gpref_smoke)
set(PYTHON_TESTS numpy_smoke)
set(MPI_TESTS mpi_smoke)
set(SILO_TESTS silo_smoke)

set(FORTRAN_TESTS fortran_smoke fruit_smoke)

################################
# Add our Main Unit Tests
################################
message(STATUS "Adding thirdparty lib unit tests")
foreach(TEST ${TPL_TESTS})
    add_cpp_test(TEST ${TEST} DEPENDS_ON libb64 )
endforeach()


################################
# Add our Optional Unit Tests
################################

if(ENABLE_GPREF_TOOLS)
    message(STATUS "Google perftools enabled: Adding related unit tests")
    foreach(TEST ${GPERF_TOOLS_TESTS})
        add_cpp_test(TEST_NAME ${TEST})
    endforeach()
else()
    message(STATUS "Google perftools disabled: Skipping related tests")
endif()

if(ENABLE_PYTHON)
    message(STATUS "Python enabled: Adding related unit tests")
    foreach(TEST ${PYTHON_TESTS})
        add_cpp_test(TEST ${TEST})
    endforeach()
else()
    message(STATUS "Python disabled: Skipping related tests")
endif()


if(ENABLE_MPI)
    message(STATUS "MPI enabled: Adding related unit tests")
    foreach(TEST ${MPI_TESTS})
        # this uses 2 procs
        add_cpp_mpi_test(TEST ${TEST} NUM_PROCS 2) 
    endforeach()
else()
    message(STATUS "MPI disabled: Skipping related tests")
endif()

if(ENABLE_SILO)
    message(STATUS "Silo enabled: Adding related unit tests")
    foreach(TEST ${SILO_TESTS})
        add_cpp_test(TEST ${TEST})
        target_link_libraries(${TEST} ${SILO_LIBRARIES})
    endforeach()
else()
    message(STATUS "Silo disabled: Skipping related tests")
endif()


if(ENABLE_FORTRAN)
    message(STATUS "Fortran enabled: Adding related unit tests")
    foreach(TEST ${FORTRAN_TESTS})
        add_fortran_test(TEST ${TEST})
    endforeach()
else()
    message(STATUS "Fortran disabled: Skipping related tests")
endif()



