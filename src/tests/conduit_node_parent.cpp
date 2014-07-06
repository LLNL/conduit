///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;


TEST(conduit_node_parent_simple, conduit_node_parent)
{
    uint32   val1  = 10;
    uint32   val2  = 20;

    Node n;
    n["a"]["b"] = val1;
    n["a"]["c"] = val2;
    EXPECT_EQ(n["a"]["b"].as_uint32(),val1);
    EXPECT_EQ(n["a"]["c"].as_uint32(),val2);
    Node &b = n["a/c"];
    EXPECT_EQ(b["../b"].as_uint32(),val1);
    EXPECT_EQ(n["a/c/../b"].as_uint32(),val1);
    
}


