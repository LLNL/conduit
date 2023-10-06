# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
###############################################################################
#
# Setup TOTALVIEW


# first Check for TOTALVIEW_DIR

if(TOTALVIEW_DIR)

  find_path(TOTALVIEW_INCLUDE_DIRECTORIES
    tv_data_display.h
    NO_DEFAULT_PATH
    PATHS ${TOTALVIEW_DIR}/include)

  find_path(TOTALVIEW_SOURCE_DIRECTORY
    tv_data_display.c
    NO_DEFAULT_PATH
    PATHS ${TOTALVIEW_DIR}/src)

  if (TOTALVIEW_INCLUDE_DIRECTORIES)
    set(TOTALVIEW_FOUND TRUE)
    set(CONDUIT_USE_TOTALVIEW TRUE)
    set(CONDUIT_EXCLUDE_TV_DATA_DISPLAY FALSE)
  endif()

endif()
