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
# Conduit Unit Tests
################################

################################
# Core (conduit lib) Unit Tests
################################
set(BASIC_TESTS gtest_smoke
                rapidjson_smoke
                conduit_smoke
                conduit_endianness
                conduit_char8_str
                conduit_datatype_tests
                conduit_node
                conduit_node_parent
                conduit_serialize
                conduit_array
                conduit_list_of
                conduit_node_binary_io
                conduit_node_save_load
                conduit_node_paths
                conduit_node_set
                conduit_to_string
                conduit_json
                conduit_json_sanitize
                conduit_generator
                conduit_node_update
                conduit_node_compact
                conduit_node_info
                conduit_node_iterator
                libb64_smoke)

################################
# conduit_io lib Unit Tests
################################
set(IO_TESTS conduit_io_smoke)

################################
# Optional Unit Tests
################################
set(PYTHON_GPREF_TOOLS_TESTS gpref_smoke)
set(PYTHON_TESTS numpy_smoke)
set(MPI_TESTS mpi_smoke conduit_mpi_smoke)
set(SILO_TESTS silo_smoke conduit_io_silo)


################################
# Add our Main Unit Tests
################################
message(STATUS "Adding conduit lib unit tests")
foreach(TEST ${BASIC_TESTS})
    add_cpp_test(TEST ${TEST} DEPENDS_ON conduit libb64 )
endforeach()

message(STATUS "Adding conduit_io lib unit tests")
foreach(TEST ${IO_TESTS})
    add_cpp_test(TEST ${TEST} DEPENDS_ON conduit conduit_io )
    if(ENABLE_SILO AND SILO_FOUND)
        target_link_libraries(${TEST} ${SILO_LIBRARIES})
    endif()
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
    add_subdirectory("python")
else()
    message(STATUS "Python disabled: Skipping related tests")
endif()


if(ENABLE_MPI)
    message(STATUS "MPI enabled: Adding related unit tests")
    foreach(TEST ${MPI_TESTS})
        add_cpp_mpi_test(TEST ${TEST} NUM_PROCS 2 DEPENDS_ON conduit conduit_mpi) # this uses 2 procs
    endforeach()
else()
    message(STATUS "MPI disabled: Skipping related tests")
endif()

if(ENABLE_SILO)
    message(STATUS "Silo enabled: Adding related unit tests")
    foreach(TEST ${SILO_TESTS})
        add_cpp_test(TEST ${TEST} DEPENDS_ON conduit conduit_io )
        if(ENABLE_SILO AND SILO_FOUND)
            target_link_libraries(${TEST} ${SILO_LIBRARIES})
        endif()
    endforeach()
else()
    message(STATUS "Silo disabled: Skipping related tests")
endif()
