###############################################################################
# Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
set(TPL_TESTS t_gtest_smoke
              t_rapidjson_smoke)

set(B64_TPL_TEST   t_libb64_smoke)
set(CIVET_TPL_TEST t_civetweb_smoke)

################################
# Optional Unit Tests
################################
set(PYTHON_TESTS      t_numpy_smoke)
set(MPI_TESTS         t_mpi_smoke)
set(HDF5_TESTS        t_hdf5_smoke)
set(SILO_TESTS        t_silo_smoke)


set(FORTRAN_TESTS t_fortran_smoke t_fruit_smoke)

################################
# Add our Main Unit Tests
################################
message(STATUS "Adding thirdparty lib unit tests")
foreach(TEST ${TPL_TESTS})
    add_cpp_test(TEST ${TEST} DEPENDS_ON)
endforeach()

message(STATUS "Adding b64 unit test")
foreach(TEST ${B64_TPL_TEST})
    add_cpp_test(TEST ${TEST}
                 SOURCES $<TARGET_OBJECTS:conduit_b64>)
endforeach()

message(STATUS "Adding civetweb unit test")


foreach(TEST ${CIVET_TPL_TEST})
    add_cpp_test(TEST ${TEST}
                 SOURCES $<TARGET_OBJECTS:conduit_civetweb>
                 DEPENDS_ON ${CIVETWEB_LIB_DEPENDS} )

    if(UNIX AND NOT APPLE)
        # we need these for civetweb on linux, we may need similar libs
        # on windows (OSX appears ok without them)
        target_link_libraries(${TEST} dl rt ${CMAKE_THREAD_LIBS_INIT})
    endif()
endforeach()


################################
# Add our Optional Unit Tests
################################

if(PYTHON_FOUND)
    message(STATUS "Python enabled: Adding related unit tests")
    foreach(TEST ${PYTHON_TESTS})
        add_cpp_test(TEST ${TEST})
    endforeach()
else()
    message(STATUS "Python disabled: Skipping related tests")
endif()


if(MPI_FOUND)
    message(STATUS "MPI enabled: Adding related unit tests")
    foreach(TEST ${MPI_TESTS})
        # this uses 2 procs
        add_cpp_mpi_test(TEST ${TEST} NUM_PROCS 2)
    endforeach()
else()
    message(STATUS "MPI disabled: Skipping related tests")
endif()


if(HDF5_FOUND)
    message(STATUS "HDF5 enabled: Adding related unit tests")
    foreach(TEST ${HDF5_TESTS})
        add_cpp_test(TEST ${TEST})
        target_link_libraries(${TEST} ${HDF5_LIBRARIES})
    endforeach()
else()
    message(STATUS "HDF5 disabled: Skipping related tests")
endif()


if(SILO_FOUND)
    message(STATUS "Silo enabled: Adding related unit tests")
    foreach(TEST ${SILO_TESTS})
        add_cpp_test(TEST ${TEST})
        target_link_libraries(${TEST} ${SILO_LIBRARIES})
    endforeach()
else()
    message(STATUS "Silo disabled: Skipping related tests")
endif()


if(FORTRAN_FOUND)
    message(STATUS "Fortran enabled: Adding related unit tests")
    foreach(TEST ${FORTRAN_TESTS})
        add_fortran_test(TEST ${TEST})
    endforeach()
else()
    message(STATUS "Fortran disabled: Skipping related tests")
endif()
