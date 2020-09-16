# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

####################################
# Add Python Module Tests
####################################
set(PYTHON_MODULE_TESTS t_python_relay_smoke
                        t_python_relay_io
                        t_python_relay_io_blueprint
                        t_python_relay_io_handle
                        t_python_relay_web)

set(PYTHON_MODULE_MPI_TESTS t_python_relay_mpi)

foreach(TEST ${PYTHON_MODULE_TESTS})
    add_python_test(TEST ${TEST}
                    FOLDER tests/relay/python)
endforeach()

if(MPI_FOUND)
    foreach(TEST ${PYTHON_MODULE_MPI_TESTS})
        add_python_mpi_test(TEST ${TEST}
                            NUM_MPI_TASKS 2)
                            # FOLDER tests/relay/python)
    endforeach()
endif()

