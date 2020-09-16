# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.


################################
# Add a staging example
################################

if(MPI_FOUND AND ENABLE_EXAMPLES)
    message(STATUS "MPI enabled: Adding staging example program")

    set(conduit_staging_support_files conduit_staging.json conduit_staging.sh)

    blt_add_executable(
        NAME        conduit_staging
        SOURCES     conduit_staging.cpp
        DEPENDS_ON  conduit conduit_relay_mpi_io 
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
        FOLDER utils)

    # Copy some helper files into the build directory.
    file(COPY ${conduit_staging_support_files}
             DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
             FILE_PERMISSIONS OWNER_READ OWNER_WRITE
                              GROUP_READ GROUP_WRITE
                              WORLD_READ)
    # add install target
    install(TARGETS conduit_staging
            RUNTIME DESTINATION bin)

    # Setup install to copy the fortran modules 
    install(FILES 
            ${conduit_staging_support_files}
            DESTINATION bin)
endif()
