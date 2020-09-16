# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# documentation targets
################################

# add a target to generate documentation with Sphinx
blt_add_sphinx_target( conduit_sphinx )
blt_set_target_folder( TARGET conduit_sphinx FOLDER docs)

if(DOXYGEN_FOUND)
   add_dependencies(conduit_sphinx conduit_doxygen)
endif()


