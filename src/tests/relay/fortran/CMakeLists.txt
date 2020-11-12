# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

###############################################################################
# Conduit Relay Fortran Interface Unit Tests
###############################################################################

set(FORTRAN_TESTS 
    t_f_relay_smoke)

################################
# Add our tests
################################
message(STATUS "Adding conduit relay lib fortran interface unit tests")
foreach(TEST ${FORTRAN_TESTS})
    add_fortran_test(TEST ${TEST}
                     DEPENDS_ON conduit conduit_relay
                     FOLDER tests/relay/fortran)
endforeach()




