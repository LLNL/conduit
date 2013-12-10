///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
// #include "rapidjson/document.h"
using namespace conduit;


TEST(conduit_ghost_node_simple_test, conduit_node)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    
    Node n;
    n.setpp<uint32>(0);
    
    GhostNode ga(&a_val, n);
    GhostNode gb(&b_val, n);
        
    EXPECT_EQ(n.getpp<uint32>(), 0);
    EXPECT_EQ(ga.getpp<uint32>(), a_val);
    EXPECT_EQ(gb.getpp<uint32>(), b_val);
}

TEST(conduit_ghost_node_object_test, conduit_node)
{
	uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);
    
    std::string schema = "{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}";
    Node n(data,schema);
    
    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);
    
    
    // Create new data, and use GhostNode on it
    a_val  = 50;
    b_val  = 60;
    c_val  = 70.0;
    
    char *data2 = new char[16];
    memcpy(&data2[0],&a_val,4);
    memcpy(&data2[4],&b_val,4);
    memcpy(&data2[8],&c_val,8);
    
    
    GhostNode g(data2, n);
   EXPECT_EQ(g["b"].getpp<uint32>(),b_val);   
   EXPECT_EQ(g["a"].getpp<uint32>(),a_val);
   EXPECT_EQ(g["b"].getpp<uint32>(),b_val);
   EXPECT_EQ(g["c"].getpp<float64>(),c_val);
}
