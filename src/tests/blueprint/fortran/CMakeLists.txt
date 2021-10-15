# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

###############################################################################
# Conduit Blueprint Fortran Interface Unit Tests
###############################################################################

set(FORTRAN_TESTS 
    t_f_blueprint_smoke
    t_f_blueprint_mcarray
    t_f_blueprint_mesh
    t_f_blueprint_table)

################################
# Add our tests
################################
message(STATUS "Adding conduit blueprint lib fortran interface unit tests")
foreach(TEST ${FORTRAN_TESTS})
    add_fortran_test(TEST ${TEST}
                     DEPENDS_ON conduit conduit_blueprint
                     FOLDER tests/blueprint/fortran)
endforeach()




