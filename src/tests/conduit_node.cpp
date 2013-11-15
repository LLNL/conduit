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

TEST(conduit_node_nested_test, conduit_node)
{

    uint32   val  = 10;

    Node n;
    n["a"]["b"] = val;
    EXPECT_EQ(n["a"]["b"].as_uint32(),val);
}

TEST(conduit_node_vec_test, conduit_node)
{

    std::vector<uint32> vec;
    for(int i=0;i<100;i++)
        vec.push_back(i);

    Node n;
    n["a"]= vec;
    EXPECT_EQ(n["a"].as_uint32_ptr()[99],99);
}



// TEST(conduit_node_simple_schema_test, conduit_node)
// {
//     uint32   a_val  = 10;
//     uint32   b_val  = 20;
//     float64  c_val  = 30.0;
// 
//     char *data = new char[16];
//     memcpy(&data[0],&a_val,4);
//     memcpy(&data[4],&b_val,4);
//     memcpy(&data[8],&c_val,8);
// 
//     Node n(data,"{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float32\"}");
// 
//     EXPECT_EQ(n["a"].as_uint32(),a_val);
//     EXPECT_EQ(n["b"].as_uint32(),b_val);
//     EXPECT_EQ(n["c"].as_float64(),c_val);
// }
