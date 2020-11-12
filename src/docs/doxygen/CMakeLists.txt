# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# documentation targets
################################

# add a target to generate API documentation with Doxygen

blt_add_doxygen_target( conduit_doxygen )
blt_set_target_folder( TARGET conduit_doxygen FOLDER docs)


