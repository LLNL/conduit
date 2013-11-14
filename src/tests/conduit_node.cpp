///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

TEST(conduit_node_simple_test, conduit_node)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);
}
