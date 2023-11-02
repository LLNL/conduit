# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

if(ENABLE_UTILS)
    blt_add_executable(
        NAME        conduit_generate_data
        SOURCES     conduit_generate_data.cpp
        OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS_ON  conduit conduit_blueprint conduit_relay
        FOLDER      utils
        )

    # add install target
    install(TARGETS conduit_generate_data
            RUNTIME DESTINATION bin)

    ################################################################
    # If we have mpi, add a parallel version.
    ################################################################

    if(MPI_FOUND)
        blt_add_executable(
            NAME        conduit_generate_data_mpi
            SOURCES     conduit_generate_data.cpp
            OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}
            DEFINES     CONDUIT_PARALLEL
            DEPENDS_ON  conduit conduit_blueprint conduit_relay conduit_relay_mpi_io ${conduit_blt_mpi_deps}
            FOLDER      utils
            )

        # add install target
        install(TARGETS conduit_generate_data_mpi
                RUNTIME DESTINATION bin)
    endif()
endif()
