#############################################################################
# Copyright (c) 2014, Lawrence Livermore National Security, LLC
# Produced at the Lawrence Livermore National Laboratory. 
# 
# All rights reserved.
# 
# This source code cannot be distributed without further review from 
# Lawrence Livermore National Laboratory.
#############################################################################
#
# Setup Silo
# This file defines:
#  SILO_FOUND - If Silo was found
#  SILO_INCLUDE_DIRS - The Silo include directories
#  SILO_LIBRARIES - The libraries needed to use Silo


# first Check for SILO_DIR, HDF5_DIR, SZIP_DIR

if(NOT SILO_DIR)
    MESSAGE(FATAL_ERROR "Silo support needs explicit SILO_DIR")
endif()

if(NOT HDF5_DIR)
    MESSAGE(FATAL_ERROR "Silo support needs explicit HDF5_DIR")
endif()

if(NOT SZIP_DIR)
    MESSAGE(FATAL_ERROR "Silo support needs explicit SZIP_DIR")
endif()


find_path(SILO_INCLUDE_DIR silo.h
          PATHS ${SILO_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)
        

find_library(SILO_LIBS NAMES siloh5
             PATHS ${SILO_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

#
# Silo Depends on libjson, hdf5 and szip:
#             

#
# libjson is part of the silo build
#
find_library(SILO_JSON_LIBS NAMES json
             PATHS ${SILO_DIR}/json/lib/
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

find_library(HDF5_LIBS NAMES hdf5 hdf5_hl
             PATHS ${HDF5_DIR}/lib/
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

find_library(SZIP_LIBS NAMES sz
             PATHS ${SZIP_DIR}/lib/
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)


set(SILO_LIBRARIES  ${SILO_LIBS} ${SILO_JSON_LIBS} ${HDF5_LIBS} ${SZIP_LIBS})
set(SILO_INCLUDE_DIRS ${SILO_INCLUDE_DIR} )


include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set SILO_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(Silo  DEFAULT_MSG
                                  SILO_LIBRARIES SILO_INCLUDE_DIRS)

mark_as_advanced(SILO_INCLUDE_DIR 
                 SILO_LIBS 
                 SILO_JSON_LIBS 
                 HDF5_LIBS 
                 SZIP_LIBS )
