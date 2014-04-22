################################
# Add our Unit Tests
################################


################################
# Core Unit Tests
################################
set(UNIT_TESTS gtest_smoke 
               rapidjson_smoke
               gpref_smoke
               conduit_smoke
               conduit_endianness
               conduit_node
               conduit_serialize
               conduit_array
               conduit_list_of
               conduit_node_paths
               conduit_to_string
               conduit_json
               conduit_io_binary
               #numpy_smoke
               libb64_smoke
               type_tests)


foreach(TEST ${UNIT_TESTS})
    add_executable(${TEST} ${TEST}.cpp )
    target_link_libraries(${TEST} gtest_main gtest conduit libb64 gperftools_lib)
    add_test( ${TEST} ${TEST} )
    # Set HEAPCHECK to local to enable explicit gpref heap checking 
    set_property(TEST ${TEST}  PROPERTY ENVIRONMENT "HEAPCHECK=local")
endforeach()

################################
# Optional Unit Tests
################################

################################
# MPI Unit Tests
################################

set(MPI_UNIT_TESTS conduit_mpi)

function(add_mpi_test TEST NUM_PROCS)
    include_directories(${MPI_CXX_INCLUDE_PATH})
    set_source_files_properties(${TEST}.cpp PROPERTIES COMPILE_FLAGS  ${MPI_CXX_COMPILE_FLAGS} )
    set_source_files_properties(${TEST}.cpp PROPERTIES LINK_FLAGS  ${MPI_CXX_LINK_FLAGS} )
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} gtest gtest_main conduit ${MPI_CXX_LIBRARIES})
    set(test_parameters ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS} "./${TEST}")
    add_test(NAME ${TEST} COMMAND ${MPIEXEC} ${test_parameters})
endfunction(add_mpi_test)

if(MPI_FOUND)
    message(STATUS "MPI found: Enabling conduit::mpi tests ")
    foreach(TEST ${MPI_UNIT_TESTS})
        add_mpi_test(conduit_mpi 2) # this uses 2 procs
    endforeach()
else()
    message(STATUS "MPI NOT found: Disabling conduit::mpi tests ")
endif()
