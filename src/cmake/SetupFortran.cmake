# Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
# Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
# other details. No copyright assignment is required to contribute to Conduit.


################################
# Guards for Fortran support.
################################
if(ENABLE_FORTRAN)
    if(CMAKE_Fortran_COMPILER)
        MESSAGE(STATUS  "Fortran Compiler: ${CMAKE_Fortran_COMPILER}")
        set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/fortran)
        
        # make sure the fortran compiler can see the module files it
        # generates
        include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})
        try_compile(Fortran_COMPILER_SUPPORTS_CLASS ${CMAKE_BINARY_DIR}
                    ${CMAKE_SOURCE_DIR}/cmake/tests/fortran_test_obj_support.f90
                    CMAKE_FLAGS "-DCMAKE_Fortran_FORMAT=FREE"
                    OUTPUT_VARIABLE OUTPUT)

        set(ENABLE_FORTRAN_OBJ_INTERFACE ${Fortran_COMPILER_SUPPORTS_CLASS})
        
    elseif(CMAKE_GENERATOR STREQUAL Xcode)
        MESSAGE(STATUS "Disabling Fortran support: ENABLE_FORTRAN is true, "
                       "but the Xcode CMake Generator does not support Fortran.")
        set(ENABLE_FORTRAN OFF)
    else()
        MESSAGE(FATAL_ERROR "ENABLE_FORTRAN is true, but a Fortran compiler wasn't found.")
    endif()
    # a this point, if ENABLE_FORTRAN is still on, we have found a Fortran compiler
    if(ENABLE_FORTRAN)
        set(FORTRAN_FOUND 1)
    endif()
endif()


