# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_relay_python_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h")
                
# Setup our modules
PYTHON_ADD_HYBRID_MODULE(conduit_relay_python
                         # dest dir
                         python-modules
                         # python module dir
                         conduit/relay
                         # python setup
                         setup.py
                         # python srcs
                         py_src/__init__.py
                         # c srcs + headers
                         conduit_relay_python.cpp
                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)

# link with the proper libs
target_link_libraries(conduit_relay_python conduit conduit_relay)

# add relay io submodule
PYTHON_ADD_COMPILED_MODULE(conduit_relay_io_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/relay/io
                           # c srcs + headers
                           conduit_relay_io_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_relay_io_python conduit conduit_relay)


# add relay web submodule
PYTHON_ADD_COMPILED_MODULE(conduit_relay_web_python
                           # dest 
                           python-modules
                           # python module dir
                           conduit/relay/web
                           # c srcs + headers
                           conduit_relay_web_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)

# link with the proper libs (beyond python)
target_link_libraries(conduit_relay_web_python conduit conduit_relay)


################################################################
# If we have mpi, add the conduit relay mpi module
################################################################

if(MPI_FOUND)

    # add relay mpi submodule
    PYTHON_ADD_COMPILED_MODULE(conduit_relay_mpi_python
                               # dest 
                               python-modules
                               # python module dir
                               conduit/relay/mpi
                               # c srcs + headers
                               conduit_relay_mpi_python.cpp
                               ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h)

    # link with the proper libs (beyond python)
    target_link_libraries(conduit_relay_mpi_python conduit conduit_relay_mpi)

endif()

