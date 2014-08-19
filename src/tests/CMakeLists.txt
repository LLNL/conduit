################################
# Conduit Unit Tests
################################

################################
# Core Unit Tests
################################
set(BASIC_TESTS gtest_smoke 
                rapidjson_smoke
                conduit_smoke
                conduit_endianness
                conduit_bytestr
                conduit_node
                conduit_node_parent
                conduit_serialize
                conduit_array
                conduit_list_of
                conduit_node_paths
                conduit_to_string
                conduit_json
                conduit_json_sanitize
                conduit_io_binary
                conduit_generator
                conduit_node_update
                conduit_node_compact
                conduit_node_info
                conduit_node_iterator
                libb64_smoke
                type_tests)

################################
# Optional Unit Tests
################################
set(PYTHON_GPREF_TOOLS_TESTS gpref_smoke)
set(PYTHON_TESTS numpy_smoke)
set(MPI_TESTS conduit_mpi)

################################
# Create the necessary lib list for our unit tests
################################
set(UNIT_TEST_LIBS gtest_main gtest conduit libb64)
if(ENABLE_GPREF_TOOLS)
    list(APPEND UNIT_TEST_LIBS gperftools_lib)
endif()

################################
# Helper to Add Unit Tests
################################
function(add_basic_test TEST)
    message(STATUS " [*] Adding Unit Test: ${TEST}")
    add_executable(${TEST} ${TEST}.cpp )
    target_link_libraries(${TEST} ${UNIT_TEST_LIBS})
    add_test( ${TEST} ${TEST} )
    if(ENABLE_GPREF_TOOLS)
        # Set HEAPCHECK to local to enable explicit gpref heap checking 
        set_property(TEST ${TEST}  PROPERTY ENVIRONMENT "HEAPCHECK=local")
    endif()
endfunction(add_basic_test)


################################
# Helper to MPI Tests
################################
function(add_mpi_test TEST NUM_PROCS)
    message(STATUS " [*] Adding MPI Unit Test: ${TEST}")
    include_directories(${MPI_CXX_INCLUDE_PATH})
    # guard against empty mpi params
    if(NOT "${MPI_CXX_COMPILE_FLAGS}" STREQUAL "")
        set_source_files_properties(${TEST}.cpp PROPERTIES COMPILE_FLAGS  ${MPI_CXX_COMPILE_FLAGS} )
    endif()
    if(NOT "${MPI_CXX_LINK_FLAGS}" STREQUAL "")
        set_source_files_properties(${TEST}.cpp PROPERTIES LINK_FLAGS  ${MPI_CXX_LINK_FLAGS} )
    endif()
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} ${UNIT_TEST_LIBS} ${MPI_CXX_LIBRARIES})
    set(test_parameters ${MPIEXEC_NUMPROC_FLAG} ${NUM_PROCS} "./${TEST}")
    add_test(NAME ${TEST} COMMAND ${MPIEXEC} ${test_parameters})
endfunction(add_mpi_test)


################################
# Add our Main Unit Tests
################################
foreach(TEST ${BASIC_TESTS})
    add_basic_test(${TEST})
endforeach()


################################
# Add our Optional Unit Tests
################################

if(ENABLE_GPREF_TOOLS)
    message(STATUS "Google perftools enabled: Adding related unit tests")
    foreach(TEST ${GPERF_TOOLS_TESTS})
        add_basic_test(${TEST})
    endforeach()
else()
    message(STATUS "Google perftools disabled: Skipping related tests")
endif()

if(ENABLE_PYTHON)
    message(STATUS "Python enabled: Adding related unit tests")
    foreach(TEST ${PYTHON_TESTS})
        add_basic_test(${TEST})
    endforeach()
else()
    message(STATUS "Python disabled: Skipping related tests")
endif()


if(ENABLE_MPI AND MPI_FOUND)
    message(STATUS "MPI enabled: Adding related unit tests")
    foreach(TEST ${MPI_TESTS})
        add_mpi_test(${TEST} 2) # this uses 2 procs
    endforeach()
else()
    message(STATUS "MPI disabled: Skipping related tests")
endif()
