///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
// #include "rapidjson/document.h"
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

TEST(conduit_node_list_test, conduit_node)
{

    std::vector<uint32> vec;
    for(int i=0;i<100;i++)
        vec.push_back(i);

    Node n;
    Node& list = n["mylist"];
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;
    list.push_back(a_val);
    list.push_back(b_val);
    list.push_back(c_val);
    list.push_back(vec);
    EXPECT_EQ(list[0].as_uint32(),a_val);
    EXPECT_EQ(list[1].as_uint32(),b_val);
    EXPECT_EQ(list[2].as_float64(),c_val);
    EXPECT_EQ(list[3].as_uint32_ptr()[99],99);

    EXPECT_EQ(n["mylist"][0].as_uint32(),a_val);
    EXPECT_EQ(n["mylist"][1].as_uint32(),b_val);
    EXPECT_EQ(n["mylist"][2].as_float64(),c_val);
    EXPECT_EQ(n["mylist"][3].as_uint32_ptr()[99],99);

}

TEST(conduit_node_simple_gen_schema_test, conduit_node)
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

    std::string schema2 = "{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}";
    Node n2(data,schema2);
    EXPECT_EQ(n2["g"]["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["g"]["b"].as_uint32(),b_val);
    EXPECT_EQ(n2["g"]["c"].as_float64(),c_val);
    
    std::string schema3 = "{\"dtype\":\"uint32\",\"length\": 5}";
    uint32 *data2 = new uint32[5];
    for (int i = 0; i < 5; i++) {
       data2[i] = i * 5;
    }
    Node n3(data2, schema3);
    for (int i = 0; i < 5; i++) {
       EXPECT_EQ(n3.as_uint32_ptr()[i], i*5);
    }
    std::string schema4 = "[\"uint32\", \"float64\", \"uint32\"]";
    char* data3 = new char[16];
    memcpy(&data3[0],&a_val,4);
    memcpy(&data3[4],&c_val,8);
    memcpy(&data3[12],&b_val,4);
    Node n4(data3, schema4);
    EXPECT_EQ(n4[0].as_uint32(),a_val);
    EXPECT_EQ(n4[1].as_float64(),c_val);
    EXPECT_EQ(n4[2].as_uint32(),b_val);

    std::string schema5 = "{\"top\":[{\"int1\":\"uint32\", \"int2\":\"uint32\"}, \"float64\", \"uint32\"], \"other\":\"float64\"}";
    char* data4 = new char[28];
    uint32   d_val  = 40;
    float64  e_val  = 50.0;
    memcpy(&data4[0],&a_val,4);
    memcpy(&data4[4],&b_val,4);
    memcpy(&data4[8],&c_val,8);
    memcpy(&data4[16],&d_val,4);
    memcpy(&data4[20],&e_val,8);
    Node n5(data4, schema5);
    EXPECT_EQ(n5["top"][0]["int1"].as_uint32(),a_val);
    EXPECT_EQ(n5["top"][0]["int2"].as_uint32(),b_val);
    EXPECT_EQ(n5["top"][1].as_float64(),c_val);
    EXPECT_EQ(n5["top"][2].as_uint32(),d_val);
    EXPECT_EQ(n5["other"].as_float64(),e_val);

}



TEST(conduit_node_simple_schema_test, conduit_node)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    n["here"]["there"] = c_val;

    std::string res = n.schema();
    std::cout << res;
    rapidjson::Document d;
    d.Parse<0>(res.c_str());

    EXPECT_TRUE(d.HasMember("a"));
    EXPECT_TRUE(d.HasMember("b"));
    EXPECT_TRUE(d.HasMember("c"));
}

TEST(conduit_node_simple_setpp_test, conduit_node)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"].setpp(a_val);
    n["b"].setpp(b_val);
    n["c"].setpp(c_val);
    
    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_float64(), c_val);
}

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


TEST(conduit_node_in_place_test, conduit_node)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;
    float64  d_val  = 40.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);
    EXPECT_EQ(*(float64*)(&data[8]), c_val);

    std::string schema = "{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}";
    Node n(data,schema);
    n["a"] = b_val;
    n["b"] = a_val;
    n["c"] = d_val;

    EXPECT_EQ(n["a"].as_uint32(), b_val);
    EXPECT_EQ(n["b"].as_uint32(), a_val);
    EXPECT_EQ(n["c"].as_float64(), d_val);

    EXPECT_EQ((uint32)(data[0]), b_val);
    EXPECT_EQ((uint32)(data[4]), a_val);
    EXPECT_EQ(*(float64*)(&data[8]), d_val);
}

