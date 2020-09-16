# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.

# Find the interpreter first
if(PYTHON_DIR AND NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${PYTHON_DIR}/bin/python)
endif()

find_package(PythonInterp REQUIRED)
if(PYTHONINTERP_FOUND)
        
        MESSAGE(STATUS "PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE}")
        
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                "import sys;from distutils.sysconfig import get_python_inc;sys.stdout.write(get_python_inc())"
                        OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
                        ERROR_VARIABLE ERROR_FINDING_INCLUDES)
        MESSAGE(STATUS "PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR}")
        
        if(NOT EXISTS ${PYTHON_INCLUDE_DIR})
            MESSAGE(FATAL_ERROR "Reported PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR} does not exist!")
        endif()

        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                "import sys;from distutils.sysconfig import get_python_lib;sys.stdout.write(get_python_lib())"
                        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES_DIR
                        ERROR_VARIABLE ERROR_FINDING_SITE_PACKAGES_DIR)
        MESSAGE(STATUS "PYTHON_SITE_PACKAGES_DIR ${PYTHON_SITE_PACKAGES_DIR}")

        if(NOT EXISTS ${PYTHON_SITE_PACKAGES_DIR})
            MESSAGE(FATAL_ERROR "Reported PYTHON_SITE_PACKAGES_DIR ${PYTHON_SITE_PACKAGES_DIR} does not exist!")
        endif()

        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                "import sys;from distutils.sysconfig import get_config_var; sys.stdout.write(get_config_var('LIBDIR'))"
                        OUTPUT_VARIABLE PYTHON_LIB_DIR
                        ERROR_VARIABLE ERROR_FINDING_LIB_DIR)
        MESSAGE(STATUS "PYTHON_LIB_DIR ${PYTHON_LIB_DIR}")

        # if we are on macOS or linux, expect PYTHON_LIB_DIR to exist
        # windows logic does not need PYTHON_LIB_DIR
        if(NOT WIN32 AND NOT EXISTS ${PYTHON_LIB_DIR})
            MESSAGE(FATAL_ERROR "Reported PYTHON_LIB_DIR ${PYTHON_LIB_DIR} does not exist!")
        endif()

        # check if we need "-undefined dynamic_lookup" by inspecting LDSHARED flags
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
                                "import sys;import sysconfig;sys.stdout.write(sysconfig.get_config_var('LDSHARED'))"
                        OUTPUT_VARIABLE PYTHON_LDSHARED_FLAGS
                        ERROR_VARIABLE ERROR_FINDING_PYTHON_LDSHARED_FLAGS)

        MESSAGE(STATUS "PYTHON_LDSHARED_FLAGS ${PYTHON_LDSHARED_FLAGS}")

        if(PYTHON_LDSHARED_FLAGS MATCHES "-undefined dynamic_lookup")
             MESSAGE(STATUS "PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG is ON")
            set(PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG ON)
        else()
             MESSAGE(STATUS "PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG is OFF")
            set(PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG OFF)
        endif()

        # check for python libs differs for windows python installs
        if(NOT WIN32)
            # we may build a shared python module against a static python
            # check for both shared and static libs cases

            # check for shared first
            set(PYTHON_GLOB_TEST "${PYTHON_LIB_DIR}/libpython*${CMAKE_SHARED_LIBRARY_SUFFIX}")
            FILE(GLOB PYTHON_GLOB_RESULT ${PYTHON_GLOB_TEST})
            # then for static if shared is not found
            if(NOT PYTHON_GLOB_RESULT)
                set(PYTHON_GLOB_TEST "${PYTHON_LIB_DIR}/libpython*${CMAKE_STATIC_LIBRARY_SUFFIX}")
            endif()
        else()
            if(PYTHON_LIB_DIR)
                set(PYTHON_GLOB_TEST "${PYTHON_LIB_DIR}/python*.lib")
            else()
                get_filename_component(PYTHON_ROOT_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
                set(PYTHON_GLOB_TEST "${PYTHON_ROOT_DIR}/libs/python*.lib")
            endif()
        endif()

        FILE(GLOB PYTHON_GLOB_RESULT ${PYTHON_GLOB_TEST})

        # make sure we found something
        if(NOT PYTHON_GLOB_RESULT)
            message(FATAL_ERROR "Failed to find main python library using pattern: ${PYTHON_GLOB_TEST}")
        endif()

        if(NOT WIN32)
            # life is ok on windows, but elsewhere
            # the glob result might be a list due to symlinks, etc
            # if it is a list, select the first entry as py lib
            list(LENGTH PYTHON_GLOB_RESULT PYTHON_GLOB_RESULT_LEN)
            if(${PYTHON_GLOB_RESULT_LEN} GREATER 1)
                list(GET PYTHON_GLOB_RESULT 0 PYTHON_GLOB_RESULT)
            endif()
        endif()

        get_filename_component(PYTHON_LIBRARY "${PYTHON_GLOB_RESULT}" ABSOLUTE)

        MESSAGE(STATUS "{PythonLibs from PythonInterp} using: PYTHON_LIBRARY=${PYTHON_LIBRARY}")
        find_package(PythonLibs)

        if(NOT PYTHONLIBS_FOUND)
            MESSAGE(FATAL_ERROR "Failed to find Python Libraries using PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}")
        endif()
        
endif()


find_package_handle_standard_args(Python  DEFAULT_MSG
                                  PYTHON_LIBRARY PYTHON_INCLUDE_DIR)



##############################################################################
# Macro to use a pure python distutils setup script
##############################################################################
FUNCTION(PYTHON_ADD_DISTUTILS_SETUP target_name
                                    dest_dir
                                    py_module_dir
                                    setup_file)
    MESSAGE(STATUS "Configuring python distutils setup: ${target_name}")
    
    # dest for build dir
    set(abs_dest_path ${CMAKE_BINARY_DIR}/${dest_dir})
    if(WIN32)
        # on windows, distutils seems to need standard "\" style paths
        string(REGEX REPLACE "/" "\\\\" abs_dest_path  ${abs_dest_path})
    endif()

    add_custom_command(OUTPUT  ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_build
            COMMAND ${PYTHON_EXECUTABLE} ${setup_file} -v
            build
            --build-base=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_build
            install
            --install-purelib="${abs_dest_path}"
            DEPENDS  ${setup_file} ${ARGN}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

    add_custom_target(${target_name} ALL DEPENDS 
                      ${CMAKE_CURRENT_BINARY_DIR}/${target_name}_build)

    # also use distutils for the install ...
    # if PYTHON_MODULE_INSTALL_PREFIX is set, install there
    if(PYTHON_MODULE_INSTALL_PREFIX)
        set(py_mod_inst_prefix ${PYTHON_MODULE_INSTALL_PREFIX})
        # make sure windows style paths don't ruin our day (or night)
        if(WIN32)
            string(REGEX REPLACE "/" "\\\\" py_mod_inst_prefix  ${PYTHON_MODULE_INSTALL_PREFIX})
        endif()
        INSTALL(CODE
            "
            EXECUTE_PROCESS(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMAND ${PYTHON_EXECUTABLE} ${setup_file} -v
                    build   --build-base=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_build_install
                    install --install-purelib=${py_mod_inst_prefix}
                OUTPUT_VARIABLE PY_DIST_UTILS_INSTALL_OUT)
            MESSAGE(STATUS \"\${PY_DIST_UTILS_INSTALL_OUT}\")
            ")
    else()
        # else install to the dest dir under CMAKE_INSTALL_PREFIX
        INSTALL(CODE
            "
            EXECUTE_PROCESS(WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMAND ${PYTHON_EXECUTABLE} ${setup_file} -v
                    build   --build-base=${CMAKE_CURRENT_BINARY_DIR}/${target_name}_build_install
                    install --install-purelib=\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${dest_dir}
                OUTPUT_VARIABLE PY_DIST_UTILS_INSTALL_OUT)
            MESSAGE(STATUS \"\${PY_DIST_UTILS_INSTALL_OUT}\")
            ")
    endif()
    
ENDFUNCTION(PYTHON_ADD_DISTUTILS_SETUP)

##############################################################################
# Macro to create a compiled python module 
##############################################################################
#
# we use this instead of the std ADD_PYTHON_MODULE cmake command 
# to setup proper install targets.
#
##############################################################################
FUNCTION(PYTHON_ADD_COMPILED_MODULE target_name
                                    dest_dir
                                    py_module_dir)
    MESSAGE(STATUS "Configuring python module: ${target_name}")
    PYTHON_ADD_MODULE(${target_name} ${ARGN})
    

    set_target_properties(${target_name} PROPERTIES
                                         LIBRARY_OUTPUT_DIRECTORY
                                         ${CMAKE_BINARY_DIR}/${dest_dir}/${py_module_dir})

    foreach(CFG_TYPE ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CFG_TYPE} CFG_TYPE)
        set_target_properties(${target_name} PROPERTIES
                                             LIBRARY_OUTPUT_DIRECTORY_${CFG_TYPE}
                                             ${CMAKE_BINARY_DIR}/${dest_dir}/${py_module_dir})
    endforeach()

    MESSAGE(STATUS "${target_name} build location: ${CMAKE_BINARY_DIR}/${dest_dir}/${py_module_dir}")

    # macOS and linux
    # defer linking with python, let the final python interpreter
    # provide the proper symbols

    # on osx we need to use the following flag to 
    # avoid undefined linking errors
    if(PYTHON_USE_UNDEFINED_DYNAMIC_LOOKUP_FLAG)
        set_target_properties(${target_name} PROPERTIES 
                              LINK_FLAGS "-undefined dynamic_lookup")
    endif()
    
    # win32, link to python
    if(WIN32)
        target_link_libraries(${target_name} ${PYTHON_LIBRARIES})
    endif()

    # support installing the python module components to an
    # an alternate dir, set via PYTHON_MODULE_INSTALL_PREFIX 
    set(py_install_dir ${dest_dir})
    if(PYTHON_MODULE_INSTALL_PREFIX)
        set(py_install_dir ${PYTHON_MODULE_INSTALL_PREFIX})
    endif()

    install(TARGETS ${target_name}
            EXPORT  conduit
            LIBRARY DESTINATION ${py_install_dir}/${py_module_dir}
            ARCHIVE DESTINATION ${py_install_dir}/${py_module_dir}
            RUNTIME DESTINATION ${py_install_dir}/${py_module_dir}
    )

ENDFUNCTION(PYTHON_ADD_COMPILED_MODULE)

##############################################################################
# Macro to create a compiled distutils and compiled python module
##############################################################################
FUNCTION(PYTHON_ADD_HYBRID_MODULE target_name
                                  dest_dir
                                  py_module_dir
                                  setup_file
                                  py_sources)
    MESSAGE(STATUS "Configuring hybrid python module: ${target_name}")

    PYTHON_ADD_DISTUTILS_SETUP("${target_name}_py_setup"
                               ${dest_dir}
                               ${py_module_dir}
                               ${setup_file}
                               ${py_sources})

    PYTHON_ADD_COMPILED_MODULE(${target_name}
                               ${dest_dir}
                               ${py_module_dir}
                               ${ARGN})

ENDFUNCTION(PYTHON_ADD_HYBRID_MODULE)


