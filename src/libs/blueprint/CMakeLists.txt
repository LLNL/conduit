# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

################################
# Add the blueprint library
################################

ENABLE_WARNINGS()


configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/conduit_blueprint_exports.h.in"
                "${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_exports.h")


#
# Specify all headers
#
set(blueprint_headers
    conduit_blueprint.hpp
    conduit_blueprint_mesh.hpp
    conduit_blueprint_mesh_utils.hpp
    conduit_blueprint_mesh_examples.hpp
    conduit_blueprint_mesh_examples_julia.hpp
    conduit_blueprint_mesh_examples_venn.hpp
    conduit_blueprint_mcarray.hpp
    conduit_blueprint_mcarray_examples.hpp
    conduit_blueprint_o2mrelation.hpp
    conduit_blueprint_o2mrelation_utils.hpp
    conduit_blueprint_o2mrelation_iterator.hpp
    conduit_blueprint_o2mrelation_examples.hpp
    conduit_blueprint_table.hpp
    conduit_blueprint_zfparray.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_exports.h
)

#
# Specify blueprint c sources
#
set(blueprint_c_headers
    c/conduit_blueprint.h
    c/conduit_blueprint_mcarray.h
    c/conduit_blueprint_mesh.h
    )


#
# Specify blueprint cpp sources
#
set(blueprint_sources
    conduit_blueprint.cpp
    conduit_blueprint_mesh.cpp
    conduit_blueprint_mesh_partition.cpp
    conduit_blueprint_mesh_utils.cpp
    conduit_blueprint_mesh_matset_xforms.cpp
    conduit_blueprint_mesh_examples.cpp
    conduit_blueprint_mesh_examples_julia.cpp
    conduit_blueprint_mesh_examples_venn.cpp
    conduit_blueprint_mcarray.cpp
    conduit_blueprint_mcarray_examples.cpp
    conduit_blueprint_o2mrelation.cpp
    conduit_blueprint_o2mrelation_utils.cpp
    conduit_blueprint_o2mrelation_iterator.cpp
    conduit_blueprint_o2mrelation_examples.cpp
    conduit_blueprint_table.cpp
    conduit_blueprint_zfparray.cpp
    )

#
# Specify blueprint c sources
#
set(blueprint_c_sources
    c/conduit_blueprint_c.cpp
    c/conduit_blueprint_mcarray_c.cpp
    c/conduit_blueprint_mesh_c.cpp
    )

#################################
# Add fortran interface
# if fortran support was selected
#################################

if(FORTRAN_FOUND)
    #
    # Specify fortran sources
    #
    set(blueprint_fortran_sources
        fortran/conduit_blueprint_fortran.f90
        fortran/conduit_blueprint_mcarray_fortran.f90
        fortran/conduit_blueprint_mesh_fortran.f90)

endif()

#
# Setup conduit blueprint lib
#
add_compiled_library(NAME conduit_blueprint
                     EXPORT conduit
                     HEADERS ${blueprint_headers} ${blueprint_c_headers}
                     SOURCES ${blueprint_sources} ${blueprint_c_sources} ${blueprint_fortran_sources}
                     DEPENDS_ON conduit
                     HEADERS_DEST_DIR include/conduit
                     FOLDER libs)


if(FORTRAN_FOUND)
    ###############################################################################
    # Special install targets for conduit fortran modules
    ###############################################################################

    set(conduit_blueprint_fortran_modules
        ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit_blueprint.mod
        ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit_blueprint_mesh.mod
        ${CMAKE_Fortran_MODULE_DIRECTORY}/conduit_blueprint_mcarray.mod)

    # Setup install to copy the fortran modules
    install(FILES
            ${conduit_blueprint_fortran_modules}
            DESTINATION include/conduit)

endif()


###################################
# add conduit_blueprint_verify exe
###################################

if(ENABLE_UTILS)
    blt_add_executable(
        NAME        conduit_blueprint_verify
        SOURCES     conduit_blueprint_verify_exe.cpp
        DEPENDS_ON  conduit conduit_relay conduit_blueprint
        OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}
        FOLDER utils)

    # add install target
    install(TARGETS conduit_blueprint_verify
            RUNTIME DESTINATION bin)
endif()

################################
# Add python wrappers if python
# support was selected
################################
if(PYTHON_FOUND)
    add_subdirectory(python)
endif()

################################################################
# If we have mpi, add the conduit blueprint mpi library
################################################################
if(MPI_FOUND)
#
# Specify blueprint mpi sources
#
set(conduit_blueprint_mpi_sources
    conduit_blueprint_mpi.cpp
    conduit_blueprint_mpi_mesh.cpp
    conduit_blueprint_mpi_mesh_partition.cpp
    conduit_blueprint_mpi_mesh_examples.cpp)

#
# Specify blueprint mpi headers
#
set(conduit_blueprint_mpi_headers
    conduit_blueprint_mpi.hpp
    conduit_blueprint_mpi_mesh.hpp
    conduit_blueprint_mpi_mesh_partition.hpp
    conduit_blueprint_mpi_mesh_examples.hpp
${CMAKE_CURRENT_BINARY_DIR}/conduit_blueprint_exports.h)

#
# Specify blueprint mpi c sources
#
set(blueprint_mpi_c_headers
    c/conduit_blueprint_mpi_mesh.h
    )

set(blueprint_mpi_c_sources
    c/conduit_blueprint_mpi_mesh_c.cpp
    )

#
# Specify blueprint mpi deps
#
set(conduit_blueprint_mpi_deps conduit
                               ${conduit_blt_mpi_deps}
                               conduit_blueprint
                               conduit_relay_mpi)

#
# setup conduit blueprint mpi lib
#
add_compiled_library(NAME conduit_blueprint_mpi
                     EXPORT conduit
                     HEADERS ${conduit_blueprint_mpi_headers} ${blueprint_mpi_c_headers}
                     SOURCES ${conduit_blueprint_mpi_sources} ${blueprint_mpi_c_sources}
                     DEPENDS_ON ${conduit_blueprint_mpi_deps}
                     HEADERS_DEST_DIR include/conduit
                     FOLDER libs)

blt_add_target_compile_flags(TO conduit_blueprint_mpi FLAGS "-DCONDUIT_BLUEPRINT_MPI_ENABLED")


endif() # end if MPI_FOUND


