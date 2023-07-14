# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
###############################################################################
#
# Example that shows how to use an installed instance of Conduit passed 
# via cmake prefix path in another CMake-based build system.
#
# To build:
#  cmake -DCMAKE_PREFIX_PATH={conduit install path} ../ -B build -S .
#  cmake --build build 
#  cd build
#  ./conduit_example
#
#
###############################################################################

cmake_minimum_required(VERSION 3.0)

project(using_with_cmake_prefix_path)

################################################################
# Import conduit using find_package search
################################################################
#
# Add Conduit's install path to CMAKE_PREFIX_PATH 
# and use following find_package call to import conduit's targets
#
find_package(Conduit REQUIRED)


######
# If Conduit was built with c++11 support, make sure we enable it
######
if(CONDUIT_USE_CXX11)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

#######################################################
# create our example 
#######################################################
add_executable(conduit_example conduit_example.cpp)

# link to conduit targets
target_link_libraries(conduit_example conduit::conduit)

# if you are using conduit's python CAPI capsule interface
# target_link_libraries(conduit_example conduit::conduit_python)


# if you are using conduit's mpi features
# if(NOT MPI_FOUND)
#    find_package(MPI COMPONENTS CXX)
# endif()
# target_link_libraries(conduit_example conduit::conduit_mpi)

