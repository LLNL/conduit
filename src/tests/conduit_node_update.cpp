///
/// file: conduit_node_update.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

TEST(update_1, conduit_node_update)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    
    Node n2;
    n2["a"] = a_val + 10;
    n2["d/aa"] = a_val;
    n2["d/bb"] = b_val;
    
    n.update(n2);
    
    EXPECT_EQ(n["a"].as_uint32(),a_val+10);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);
    EXPECT_EQ(n["d/aa"].as_uint32(),a_val);
    EXPECT_EQ(n["d/bb"].as_uint32(),b_val);
}
