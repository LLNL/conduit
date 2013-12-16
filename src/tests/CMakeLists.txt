################################
# Add our Unit Tests
################################

set(UNIT_TESTS gtest_smoke 
               rapidjson_smoke
               conduit_smoke
               conduit_endianness
               conduit_node
               conduit_serialize
               conduit_array
               conduit_node_paths
               conduit_ghost_node
               numpy_smoke
               type_tests)

foreach(TEST ${UNIT_TESTS})
    add_executable(${TEST} ${TEST}.cpp )
    target_link_libraries(${TEST} gtest gtest_main conduit)
    add_test( ${TEST} ${TEST} )
endforeach()
