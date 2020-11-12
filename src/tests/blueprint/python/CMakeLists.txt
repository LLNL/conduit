# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

####################################
# Add Python Module Tests
####################################
set(PYTHON_MODULE_TESTS t_python_blueprint_smoke
                        t_python_blueprint_mcarray
                        t_python_blueprint_mesh)


foreach(TEST ${PYTHON_MODULE_TESTS})
    add_python_test(TEST ${TEST} FOLDER tests/blueprint/python)
endforeach()

