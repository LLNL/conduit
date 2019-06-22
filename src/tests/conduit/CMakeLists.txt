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

################################
# Conduit Unit Tests
################################

################################
# Core (conduit lib) Unit Tests
################################
set(BASIC_TESTS t_conduit_smoke
                t_conduit_endianness
                t_conduit_char8_str
                t_conduit_datatype_tests
                t_conduit_node
                t_conduit_node_parent
                t_conduit_node_compare
                t_conduit_serialize
                t_conduit_array
                t_conduit_list_of
                t_conduit_node_binary_io
                t_conduit_node_save_load
                t_conduit_node_paths
                t_conduit_node_set
                t_conduit_node_to_value
                t_conduit_node_to_array
                t_conduit_to_string
                t_conduit_json
                t_conduit_json_sanitize
                t_conduit_yaml
                t_conduit_generator
                t_conduit_node_update
                t_conduit_node_compact
                t_conduit_node_info
                t_conduit_node_iterator
                t_conduit_schema
                t_conduit_error
                t_conduit_log
                t_conduit_utils)


################################
# Add our tests
################################
message(STATUS "Adding conduit lib unit tests")
foreach(TEST ${BASIC_TESTS})
    add_cpp_test(TEST ${TEST}
                 DEPENDS_ON conduit
                 FOLDER tests/conduit)
endforeach()

################################
# Add c interface tests
################################
add_subdirectory("c")

################################
# Add optional tests
################################
if(PYTHON_FOUND)
    add_subdirectory("python")
else()
    message(STATUS "Python disabled: Skipping conduit python module tests")
endif()

if(FORTRAN_FOUND)
    add_subdirectory("fortran")
else()
    message(STATUS "Fortran disabled: Skipping conduit fortran interface tests")
endif()


