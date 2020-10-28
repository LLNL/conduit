# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_python_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_python_exports.h")

# Specify the sources of the pure python and compiled portions of our module.
SET(conduit_py_python_sources  py_src/__init__.py)

SET(conduit_py_headers         conduit_python.hpp
                               ${CMAKE_CURRENT_BINARY_DIR}/conduit_python_exports.h)

SET(conduit_py_cpp_sources     conduit_python.cpp)

# Make sure we have access to numpy
include_directories(${NUMPY_INCLUDE_DIRS})

# add conduit + conduit_python modules
PYTHON_ADD_HYBRID_MODULE(NAME          conduit_python
                         DEST_DIR      python-modules
                         PY_MODULE_DIR conduit
                         PY_SETUP_FILE setup.py
                         PY_SOURCES    ${conduit_py_python_sources}
                         SOURCES       ${conduit_py_headers}
                                       ${conduit_py_cpp_sources}
                         FOLDER        libs/python)

# link with the proper libs (beyond python)
target_link_libraries(conduit_python conduit)

# install the capi header so other python modules can use it
# support alt install dir for python module via PYTHON_MODULE_INSTALL_PREFIX
if(PYTHON_MODULE_INSTALL_PREFIX)
    install(FILES ${conduit_py_headers} DESTINATION ${PYTHON_MODULE_INSTALL_PREFIX}/conduit/)
else()
    install(FILES ${conduit_py_headers} DESTINATION python-modules/conduit/)
endif()


# create scripts that make it easy to launch the proper 
# python interp with the conduit module already in PYTHONPATH

set(_PYMOD_DIR ${PROJECT_BINARY_DIR}/python-modules)
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/run_python_with_conduit.sh.in"
                "${CMAKE_CURRENT_BINARY_DIR}/run_python_with_conduit_build.sh" @ONLY)

if(PYTHON_MODULE_INSTALL_PREFIX)
    set(_PYMOD_DIR ${PYTHON_MODULE_INSTALL_PREFIX})
else()
    set(_PYMOD_DIR ${CMAKE_INSTALL_PREFIX}/python-modules/)
endif()


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/run_python_with_conduit.sh.in"
                "${CMAKE_CURRENT_BINARY_DIR}/run_python_with_conduit_install.sh" @ONLY)

unset(_PYMOD_DIR)

# install the script that works for install paths
# 'PROGRAMS' makes sure exec perms are set
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/run_python_with_conduit_install.sh"
        DESTINATION bin
        RENAME run_python_with_conduit.sh)






