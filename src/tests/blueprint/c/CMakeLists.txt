# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

###############################################################################
# Conduit Blueprint C Interface Unit Tests
###############################################################################

set(BASIC_C_TESTS
     t_c_blueprint_smoke
     t_c_blueprint_mcarray
     t_c_blueprint_mesh
     t_c_blueprint_table)

################################
# Add our tests
################################
message(STATUS "Adding conduit blueprint lib c interface unit tests")
foreach(TEST ${BASIC_C_TESTS})
    add_cpp_test(TEST ${TEST}
                 DEPENDS_ON conduit conduit_blueprint
                 FOLDER tests/blueprint/c)
endforeach()


