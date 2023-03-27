# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

if(ENABLE_UTILS)
    blt_add_executable(
        NAME        conduit_verify_adjset
        SOURCES     adjset_verify.cpp
        OUTPUT_DIR  ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS_ON  conduit conduit_blueprint conduit_relay
        FOLDER      utils
        )

    # add install target
    install(TARGETS conduit_verify_adjset
            RUNTIME DESTINATION bin)
endif()
