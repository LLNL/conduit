###############################################################################
# Copyright (c) Copyright (c) 2015-2019, Lawrence Livermore National Security, LLC.
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
# Unit Tests for 3rd Party Libs
################################

message(STATUS "Adding thirdparty lib unit tests")


add_cpp_test(TEST t_rapidjson_smoke
             FOLDER tests/thirdparty)

add_cpp_test(TEST t_libb64_smoke
                 SOURCES $<TARGET_OBJECTS:conduit_b64>
                 FOLDER tests/thirdparty)


add_cpp_test(TEST t_libyaml_smoke
            SOURCES $<TARGET_OBJECTS:conduit_libyaml>
            FOLDER tests/thirdparty)

add_cpp_test(TEST t_fmt_smoke
             FOLDER tests/thirdparty)


set(civet_test_deps ${CIVETWEB_LIB_DEPENDS})

if(UNIX AND NOT APPLE)
    # we need these for civetweb on linux, we may need similar libs
    # on windows (OSX appears ok without them)
    list(APPEND civet_test_deps dl rt ${CMAKE_THREAD_LIBS_INIT})
endif()


if (MINGW)
  list(APPEND civet_test_deps ws2_32)
endif ()

add_cpp_test(TEST t_civetweb_smoke
             SOURCES $<TARGET_OBJECTS:conduit_civetweb>
             DEPENDS_ON ${civet_test_deps}
             FOLDER tests/thirdparty)

################################
# Optional TPL Unit Tests
################################

if(PYTHON_FOUND)
    message(STATUS "Python enabled: Adding related unit tests")
    foreach(TEST t_numpy_smoke)
        add_cpp_test(TEST t_numpy_smoke
                     FOLDER tests/thirdparty)
        target_link_libraries(${TEST} PUBLIC ${PYTHON_LIBRARIES})
    endforeach()
else()
    message(STATUS "Python disabled: Skipping related tests")
endif()


if(HDF5_FOUND)
    message(STATUS "HDF5 enabled: Adding related unit tests")
    if(HDF5_IS_PARALLEL AND ENABLE_MPI)
       set(HDF5_MPI_DEP mpi)
    endif()
    add_cpp_test(TEST t_hdf5_smoke
                 DEPENDS_ON hdf5 ${HDF5_MPI_DEP}
                 FOLDER tests/thirdparty)
else()
    message(STATUS "HDF5 disabled: Skipping related tests")
endif()


if(SILO_FOUND)
    if(HDF5_FOUND AND HDF5_IS_PARALLEL AND ENABLE_MPI)
       set(SILO_HDF5_MPI_DEP mpi)
    endif()
    message(STATUS "Silo enabled: Adding related unit tests")
        add_cpp_test(TEST t_silo_smoke
                     DEPENDS_ON silo ${SILO_HDF5_MPI_DEP}
                     FOLDER tests/thirdparty)
else()
    message(STATUS "Silo disabled: Skipping related tests")
endif()

if(ADIOS_FOUND)
    set(serial_adios_deps adios_nompi)
    if(UNIX AND NOT APPLE)
        list(APPEND serial_adios_deps dl rt ${CMAKE_THREAD_LIBS_INIT})
    endif()
    message(STATUS "ADIOS enabled: Adding related unit tests")
        add_cpp_test(TEST t_adios_smoke
                     DEPENDS_ON ${serial_adios_deps}
                     FOLDER tests/thirdparty)
else()
    message(STATUS "ADIOS disabled: Skipping related tests")
endif()

if(FORTRAN_FOUND)
    message(STATUS "Fortran enabled: Adding related unit tests")
    add_fortran_test(TEST t_fortran_smoke)
else()
    message(STATUS "Fortran disabled: Skipping related tests")
endif()
