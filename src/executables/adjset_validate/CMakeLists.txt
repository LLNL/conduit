# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

if(ENABLE_UTILS)
    blt_add_executable(
        NAME        conduit_adjset_validate
        SOURCES     adjset_validate.cpp
        OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS_ON  conduit conduit_blueprint conduit_relay
        FOLDER      utils
        )

    # add install target
    install(TARGETS conduit_adjset_validate
            RUNTIME DESTINATION bin)
endif()
