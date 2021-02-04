# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

####################################
# Add Python Module Tests
####################################
set(PYTHON_MODULE_TESTS t_python_conduit_smoke
                        t_python_conduit_node
                        t_python_conduit_utils
                        t_python_conduit_schema
                        t_python_conduit_datatype
                        t_python_conduit_generator
                        t_python_conduit_node_iterator
                        t_python_conduit_endianness)


foreach(TEST ${PYTHON_MODULE_TESTS})
    add_python_test(TEST ${TEST}
                    FOLDER tests/conduit/python)
endforeach()

