# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

include(CMakeFindDependencyMacro)

###############################################################################
# Setup Threads
###############################################################################
if(UNIX AND NOT APPLE)
# we depend on Threads::Threads in our exported targets
# so we need to bootstrap that here
    if(NOT Threads::Threads)
        find_package( Threads REQUIRED )
    endif()
endif()

###############################################################################
# Setup HDF5
###############################################################################
if(CONDUIT_HDF5_DIR)
    message(STATUS "Conduit was built with HDF5 Support")
    # we depend on hdf5 in our exported targets
    # If ZZZ_DIR not set, use known install path for HDF5
    if(NOT HDF5_DIR)
        # if conduit was configured with hdf, we need to include it
        set(HDF5_DIR ${CONDUIT_HDF5_DIR})
    endif()

    # this logic mirrors Conduit's SetupHDF5.cmake

    # find the absolute path w/ symlinks resolved of the passed HDF5_DIR, 
    # since sanity checks later need to compare against the real path
    get_filename_component(HDF5_DIR_REAL "${HDF5_DIR}" REALPATH)
    message(STATUS "Looking for HDF5 at: " ${HDF5_DIR_REAL})

    if(POLICY CMP0074)
        #policy for <PackageName>_ROOT variables
        cmake_policy(PUSH)
        cmake_policy(SET CMP0074 NEW)
    endif()

    # CMake's FindHDF5 module uses the HDF5_ROOT env var
    set(HDF5_ROOT ${HDF5_DIR_REAL})

    if(NOT WIN32)
        # use HDF5_ROOT env var for FindHDF5 with older versions of cmake
        if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
            set(ENV{HDF5_ROOT} ${HDF5_ROOT}/bin)
        endif()

        # Use CMake's FindHDF5 module to locate hdf5 and setup hdf5
        find_package(HDF5 REQUIRED)
    else()
        # CMake's FindHDF5 module is buggy on windows and will put the dll
        # in HDF5_LIBRARY.  Instead, use the 'CONFIG' signature of find_package
        # with appropriate hints for where cmake can find hdf5-config.cmake.
        find_package(HDF5 CONFIG 
                     REQUIRED
                     HINTS ${HDF5_DIR}/cmake/hdf5 
                           ${HDF5_DIR}/lib/cmake/hdf5
                           ${HDF5_DIR}/share/cmake/hdf5
                           ${HDF5_DIR}/cmake)
    endif()

    if(POLICY CMP0074)
        # clear CMP0074
        cmake_policy(POP)
    endif()

    # FindHDF5/find_package sets HDF5_DIR to it's installed CMake info if it exists
    # we want to keep HDF5_DIR as the root dir of the install to be 
    # consistent with other packages

    # find the absolute path w/ symlinks resolved of the passed HDF5_DIR, 
    # since sanity checks later need to compare against the real path
    get_filename_component(HDF5_DIR_REAL "${HDF5_ROOT}" REALPATH)

    set(HDF5_DIR ${HDF5_DIR_REAL} CACHE PATH "" FORCE)
    message(STATUS "HDF5_DIR_REAL=${HDF5_DIR_REAL}")
    #
    # Sanity check to alert us if some how we found an hdf5 instance
    # in an unexpected location.
    #
    message(STATUS "Checking that found HDF5_INCLUDE_DIRS are in HDF5_DIR")

    #
    # HDF5_INCLUDE_DIRS may also include paths to external lib headers 
    # (such as szip), so we check that *at least one* of the includes
    # listed in HDF5_INCLUDE_DIRS exists in the HDF5_DIR specified.
    #

    # HDF5_INCLUDE_DIR is deprecated, but there are still some cases
    # where HDF5_INCLUDE_DIR is set, but HDF5_INCLUDE_DIRS is not
    if(NOT HDF5_INCLUDE_DIRS)
        if(HDF5_INCLUDE_DIR)
            set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
        else()
            message(FATAL_ERROR "FindHDF5 did not provide HDF5_INCLUDE_DIRS or HDF5_INCLUDE_DIR.")
        endif()
    endif()

    message(STATUS "HDF5_INCLUDE_DIRS=${HDF5_INCLUDE_DIRS}")
    set(check_hdf5_inc_dir_ok 0)
    foreach(IDIR ${HDF5_INCLUDE_DIRS})

        # get real path of the include dir
        # w/ abs and symlinks resolved
        get_filename_component(IDIR_REAL "${IDIR}" REALPATH)
        # check if idir_real is a substring of hdf5_dir

        if("${IDIR_REAL}" MATCHES "${HDF5_DIR}")
            message(STATUS " ${IDIR_REAL} includes HDF5_DIR (${HDF5_DIR})")
            set(check_hdf5_inc_dir_ok 1)
        endif()

        if("${IDIR_REAL}" MATCHES "${HDF5_REAL_DIR}")
            message(STATUS " ${IDIR_REAL} includes HDF5_REAL_DIR (${HDF5_REAL_DIR})")
            set(check_hdf5_inc_dir_ok 1)
        endif()

    endforeach()

    if(NOT check_hdf5_inc_dir_ok)
        message(FATAL_ERROR " ${HDF5_INCLUDE_DIRS} does not include HDF5_DIR")
    endif()

    #
    # filter HDF5_LIBRARIES to remove hdf5_hl if it exists
    # we don't use hdf5_hl, but if we link with it will become
    # a transitive dependency
    #
    set(HDF5_HL_LIB FALSE)
    foreach(LIB ${HDF5_LIBRARIES})
        if("${LIB}" MATCHES "hdf5_hl")
            set(HDF5_HL_LIB ${LIB})
        endif()
    endforeach()

    if(HDF5_HL_LIB)
        message(STATUS "Removing hdf5_hl from HDF5_LIBRARIES")
        list(REMOVE_ITEM HDF5_LIBRARIES ${HDF5_HL_LIB})
    endif()

else()
    message(STATUS "Conduit was NOT built with HDF5 Support")
endif()


