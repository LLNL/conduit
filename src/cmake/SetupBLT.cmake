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

if(ENABLE_MPI)
    # adjust MPI from BLT
    if( ${CMAKE_VERSION} VERSION_LESS "3.15.0" )
        # older cmake, we use BLT's mpi support, it uses 
        # the name mpi
        set(conduit_blt_mpi_deps mpi CACHE STRING "")
    else()
        if(TARGET MPI::MPI_CXX)
            message(STATUS "Using MPI CMake imported target: MPI::MPI_CXX")
            # newer cmake we use find mpi targets directly
            set(conduit_blt_mpi_deps MPI::MPI_CXX CACHE STRING "")
        else()
            message(FATAL_ERROR "Cannot use CMake imported targets for MPI."
                                "(CMake > 3.15, ENABLE_MPI == ON, but "
                                "MPI::MPI_CXX CMake target is missing.)")
        endif()
    endif()
endif()

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
