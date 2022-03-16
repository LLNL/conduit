# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_relay_python_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h")


# add conduit.relay + conduit_relay_python modules
# PYTHON_ADD_HYBRID_MODULE(NAME          conduit_relay_python
#                          DEST_DIR      python-modules
#                          PY_MODULE_DIR conduit/relay
#                          PY_SETUP_FILE setup.py
#                          PY_SOURCES    py_src/__init__.py
#                                        py_src/io/__init__.py
#                                        py_src/io/blueprint/__init__.py
#                                        py_src/mpi/__init__.py
#                                        py_src/web/__init__.py
#                          SOURCES       conduit_relay_python.cpp
#                                        ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h
#                          FOLDER        libs/python)

# add relay submodule
PYTHON_ADD_COMPILED_MODULE(NAME          conduit_relay_python
                           DEST_DIR      python-modules
                           PY_MODULE_DIR conduit/relay
                           SOURCES       conduit_relay_python.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h
                           FOLDER        libs/python)

# link with the proper libs
target_link_libraries(conduit_relay_python conduit conduit_relay)

# add relay io submodule
PYTHON_ADD_COMPILED_MODULE(NAME          conduit_relay_io_python
                           DEST_DIR      python-modules
                           PY_MODULE_DIR conduit/relay/io
                           SOURCES       conduit_relay_io_python.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h
                           FOLDER        libs/python)


# link with the proper libs (beyond python)
target_link_libraries(conduit_relay_io_python conduit conduit_relay)

# add relay io blueprint submodule
PYTHON_ADD_COMPILED_MODULE(NAME          conduit_relay_io_blueprint_python
                           DEST_DIR      python-modules
                           PY_MODULE_DIR conduit/relay/io/blueprint
                           SOURCES       conduit_relay_io_blueprint_python.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h
                           FOLDER        libs/python)

# link with the proper libs (beyond python)
target_link_libraries(conduit_relay_io_blueprint_python conduit conduit_relay conduit_blueprint)


if(ENABLE_RELAY_WEBSERVER)
    # add relay web submodule
    PYTHON_ADD_COMPILED_MODULE(NAME          conduit_relay_web_python
                               DEST_DIR      python-modules 
                               PY_MODULE_DIR conduit/relay/web
                               SOURCES       conduit_relay_web_python.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h
                               FOLDER        libs/python)

    # link with the proper libs (beyond python)
    target_link_libraries(conduit_relay_web_python conduit conduit_relay)
endif()

################################################################
# If we have mpi, add the conduit relay mpi module
################################################################

if(MPI_FOUND)

    # add relay mpi submodule
    PYTHON_ADD_COMPILED_MODULE(NAME          conduit_relay_mpi_python
                               DEST_DIR      python-modules
                               PY_MODULE_DIR conduit/relay/mpi
                               SOURCES       conduit_relay_mpi_python.cpp
                                             ${CMAKE_CURRENT_BINARY_DIR}/conduit_relay_python_exports.h
                               FOLDER        libs/python)

    # link with the proper libs (beyond python)
    target_link_libraries(conduit_relay_mpi_python conduit conduit_relay_mpi)

endif()

