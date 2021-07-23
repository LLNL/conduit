# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.
#
# Setup ParMeTiS
# This file defines:
#  PARMETIS_DIRFOUND - If ParMeTiS was found
#  PARMETIS_DIR_INCLUDE_DIRS - The ParMeTiS include directories
#  PARMETIS_DIR_LIBRARIES - The libraries needed to use ParMeTiS


# first Check for PARMETIS_DIR

if(NOT PARMETIS_DIR)
    MESSAGE(FATAL_ERROR "Parmetis support needs explicit PARMETIS_DIR")
endif()


find_path(PARMETIS_INCLUDE_DIR parmetis.h
          PATHS ${PARMETIS_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)
        

find_library(PARMETIS_LIB NAMES parmetis
             PATHS ${PARMETIS_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

set(PARMETIS_LIBRARIES  ${PARMETIS_LIB})
set(PARMETIS_INCLUDE_DIRS ${PARMETIS_INCLUDE_DIR} )


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set PARMETIS_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Parmetis  DEFAULT_MSG
                                  PARMETIS_LIBRARIES PARMETIS_INCLUDE_DIRS)

mark_as_advanced(PARMETIS_INCLUDE_DIR 
                 PARMETIS_LIB)


blt_register_library(NAME parmetis
                     INCLUDES ${PARMETIS_INCLUDE_DIRS}
                     LIBRARIES ${PARMETIS_LIBRARIES} )

