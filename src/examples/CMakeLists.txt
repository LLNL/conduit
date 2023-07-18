# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

###############################################################################
#
# file: src/examples/CMakeLists.txt
#
###############################################################################

add_subdirectory(staging)

# install using and python examples
install(DIRECTORY using-with-cmake
                  using-with-cmake-mpi
                  using-with-cmake-prefix-path
                  using-with-make
                  python
                  cpp_fort_and_py
        DESTINATION examples/conduit)
