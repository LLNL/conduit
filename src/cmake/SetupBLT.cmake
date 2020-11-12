# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################################################
# if BLT_SOURCE_DIR is not set - use "blt" as default
################################################################
if(NOT BLT_SOURCE_DIR)
    set(BLT_SOURCE_DIR "blt")
endif()

################################################################
# if not set, prefer c++11 lang standard
################################################################
if(NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++11" CACHE STRING "")
endif()

################################################################
# if not set, prefer folder grouped targets
################################################################
if(NOT ENABLE_FOLDERS)
    set(ENABLE_FOLDERS TRUE CACHE STRING "")
endif()

################################################################
# init blt using BLT_SOURCE_DIR
################################################################
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)


################################################################
# apply folders to a few ungrouped blt targets
################################################################

###############################################
# group main blt docs targets into docs folder
###############################################
blt_set_target_folder( TARGET docs FOLDER docs)

if(TARGET sphinx_docs)
    blt_set_target_folder( TARGET sphinx_docs FOLDER docs)
endif()

if(TARGET doxygen_docs)
    blt_set_target_folder( TARGET doxygen_docs FOLDER docs)
endif()

####################################################
# group top level blt health checks into blt folder
####################################################
if(TARGET check)
    blt_set_target_folder( TARGET check FOLDER blt)
endif()

if(TARGET style)
    blt_set_target_folder( TARGET style FOLDER blt)
endif()
