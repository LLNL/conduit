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

###############################################################################
# Conduit Relay C Interface Unit Tests
###############################################################################

set(RELAY_C_TESTS
     t_c_relay_smoke)

set(RELAY_C_IO_ADIOS_TESTS t_c_relay_io_adios)
set(RELAY_C_MPI_IO_ADIOS_TESTS t_c_relay_mpi_io_adios)

################################
# Add our tests
################################
message(STATUS "Adding conduit relay lib c interface unit tests")
foreach(TEST ${RELAY_C_TESTS})
    add_cpp_test(TEST ${TEST} DEPENDS_ON conduit conduit_relay)
endforeach()


if(ADIOS_FOUND)
    message(STATUS "ADIOS enabled: Adding conduit_relay c interface ADIOS unit tests")
    foreach(TEST ${RELAY_C_IO_ADIOS_TESTS})
        add_cpp_test(TEST ${TEST} DEPENDS_ON conduit conduit_relay)
    endforeach()

    if(MPI_FOUND)
        message(STATUS "MPI enabled: Adding conduit_relay_mpi_adios unit tests")
        foreach(TEST ${RELAY_C_MPI_IO_ADIOS_TESTS})
            add_cpp_mpi_test(TEST ${TEST} NUM_MPI_TASKS 2 DEPENDS_ON conduit conduit_relay_mpi_io) 
        endforeach()
    endif()
else()
    message(STATUS "ADIOS disabled: Skipping conduit_relay c interface ADIOS tests")
endif()


