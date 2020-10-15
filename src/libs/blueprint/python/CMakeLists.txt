# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_blueprint_python_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h")


# add conduit.blueprint + conduit_blueprint_python modules
PYTHON_ADD_HYBRID_MODULE(NAME          conduit_blueprint_python
                         DEST_DIR      python-modules
                         PY_MODULE_DIR conduit/blueprint
                         PY_SETUP_FILE setup.py
                         PY_SOURCES    py_src/__init__.py
                         SOURCES       conduit_blueprint_python.cpp
                                       ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_python conduit conduit_blueprint)

#############################################################
# blueprint.mcarray
#############################################################

# add mcarray submodule
PYTHON_ADD_COMPILED_MODULE(NAME          conduit_blueprint_mcarray_python
                           DEST_DIR      python-modules
                           PY_MODULE_DIR conduit/blueprint/mcarray
                           SOURCES       conduit_blueprint_mcarray_python.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)

# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mcarray_python conduit conduit_blueprint)

# add mcarray examples submodule
PYTHON_ADD_COMPILED_MODULE(NAME          conduit_blueprint_mcarray_examples_python
                           DEST_DIR      python-modules
                           PY_MODULE_DIR conduit/blueprint/mcarray/examples
                           SOURCES       conduit_blueprint_mcarray_examples_python.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)

# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mcarray_examples_python conduit conduit_blueprint)

#############################################################
# blueprint.mesh
#############################################################

# add mesh submodule
PYTHON_ADD_COMPILED_MODULE(NAME          conduit_blueprint_mesh_python
                           DEST_DIR      python-modules
                           PY_MODULE_DIR conduit/blueprint/mesh
                           SOURCES       conduit_blueprint_mesh_python.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)

# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mesh_python conduit conduit_blueprint)


# add mesh examples submodule
PYTHON_ADD_COMPILED_MODULE(NAME          conduit_blueprint_mesh_examples_python
                           DEST_DIR      python-modules
                           PY_MODULE_DIR conduit/blueprint/mesh/examples
                           SOURCES       conduit_blueprint_mesh_examples_python.cpp
                                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)

# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mesh_examples_python conduit conduit_blueprint)


