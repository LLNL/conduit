# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_blueprint_python_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h")


# Setup our module
PYTHON_ADD_HYBRID_MODULE(conduit_blueprint_python
                         python-modules
                         conduit/blueprint
                         setup.py
                         # python srcs
                         py_src/__init__.py
                         # c srcs + headers
                         conduit_blueprint_python.cpp
                         ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)

# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_python conduit conduit_blueprint)

#############################################################
# blueprint.mcarray
#############################################################

# add mcarray submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mcarray_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mcarray
                           # c srcs + headers
                           conduit_blueprint_mcarray_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mcarray_python conduit conduit_blueprint)

# add mcarray examples submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mcarray_examples_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mcarray/examples
                           # c srcs + headers
                           conduit_blueprint_mcarray_examples_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mcarray_examples_python conduit conduit_blueprint)

#############################################################
# blueprint.mesh
#############################################################

# add mesh submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mesh_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mesh
                           # c srcs + headers
                           conduit_blueprint_mesh_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mesh_python conduit conduit_blueprint)


# add mesh examples submodule
PYTHON_ADD_COMPILED_MODULE(conduit_blueprint_mesh_examples_python
                           # dest_dir 
                           python-modules
                           # python module dir
                           conduit/blueprint/mesh/examples
                           # c srcs + headers
                           conduit_blueprint_mesh_examples_python.cpp
                           ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_python_exports.h)


# link with the proper libs (beyond python)
target_link_libraries(conduit_blueprint_mesh_examples_python conduit conduit_blueprint)


