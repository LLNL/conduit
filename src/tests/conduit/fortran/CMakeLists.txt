# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

###############################################################################
# Conduit Fortran Interface Unit Tests
###############################################################################

set(FORTRAN_TESTS 
    t_f_type_sizes
    t_f_conduit_smoke
    t_f_conduit_node
    t_f_conduit_node_int
    t_f_conduit_node_float
    t_f_conduit_node_char8_str)

if(ENABLE_FORTRAN_OBJ_INTERFACE)
    list(APPEND FORTRAN_TESTS t_f_conduit_node_obj)
endif()

################################
# Add our tests
################################
message(STATUS "Adding conduit lib fortran interface unit tests")
foreach(TEST ${FORTRAN_TESTS})
    add_fortran_test(TEST ${TEST}
                     DEPENDS_ON conduit
                     FOLDER tests/conduit/fortran)
endforeach()




