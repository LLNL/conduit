# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# documentation targets
################################

# add a targets to generate API documentation with doxygen 
if(DOXYGEN_FOUND)
    add_subdirectory(doxygen)
endif()

# add a targets to generate user and developer documentation with sphinx
if(SPHINX_FOUND)
    add_subdirectory(sphinx)
endif()


