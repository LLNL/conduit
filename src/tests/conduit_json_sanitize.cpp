///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

TEST(conduit_json_sanitize_comments, conduit_json_sanitize)
{
    std::string t1_in  = "//comment\n{\"a\": \"uint32\", \"b\": \"uint32\" , \"c\": \"float64\"}\n// comment!";
    std::string t1_out = "{\"a\": \"uint32\", \"b\": \"uint32\" , \"c\": \"float64\"}\n";
    
    EXPECT_EQ(utils::json_sanitize(t1_in),t1_out);
}


TEST(conduit_json_sanitize_quoteless, conduit_json_sanitize)
{
    std::string t1_in  = "{a: uint32, b: uint32 , c: float64}";
    std::string t1_out = "{\"a\": \"uint32\", \"b\": \"uint32\" , \"c\": \"float64\"}";

    std::string t2_in  = "{g: {a: uint32, b: uint32 , c: float64}}";
    std::string t2_out = "{\"g\": {\"a\": \"uint32\", \"b\": \"uint32\" , \"c\": \"float64\"}}";
    
    std::string t3_in  = "{dtype: uint32, length: 5}";
    std::string t3_out = "{\"dtype\": \"uint32\", \"length\": 5}";

    std::string t4_in  = "[ uint32, float64, uint32]";
    std::string t4_out = "[ \"uint32\", \"float64\", \"uint32\"]";
    
    std::string t5_in  = "{top:[ {int1: uint32, int2: uint32}, float64, uint32], other: float64}";
    std::string t5_out = "{\"top\":[ {\"int1\": \"uint32\", \"int2\": \"uint32\"}, \"float64\", \"uint32\"], \"other\": \"float64\"}";
    
    EXPECT_EQ(utils::json_sanitize(t1_in),t1_out);
    EXPECT_EQ(utils::json_sanitize(t2_in),t2_out);
    EXPECT_EQ(utils::json_sanitize(t3_in),t3_out);
    EXPECT_EQ(utils::json_sanitize(t4_in),t4_out);
    EXPECT_EQ(utils::json_sanitize(t5_in),t5_out);
}

TEST(conduit_node_quoteless_simple_gen_schema_test, conduit_node_quoteless)
{
    return;
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);

    Schema schema("{a: uint32, b: uint32 , c: float64}");
    Node n(schema,data);
    
    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);

    std::string s2_str("{g: {a: uint32, b: uint32 , c: float64}}");
    std::cout << s2_str << std::endl;
    Schema schema2(s2_str);
    
    Node n2(schema2,data);
    EXPECT_EQ(n2["g"]["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["g"]["b"].as_uint32(),b_val);
    EXPECT_EQ(n2["g"]["c"].as_float64(),c_val);
    
    Schema schema3("{dtype: uint32, length: 5}");
    uint32 *data2 = new uint32[5];
    for (int i = 0; i < 5; i++) {
       data2[i] = i * 5;
    }
    Node n3(schema3,data2);
    for (int i = 0; i < 5; i++) {
       EXPECT_EQ(n3.as_uint32_ptr()[i], i*5);
    }
    Schema schema4("[ uint32, float64, uint32]");
    char* data3 = new char[16];
    memcpy(&data3[0],&a_val,4);
    memcpy(&data3[4],&c_val,8);
    memcpy(&data3[12],&b_val,4);
    Node n4(schema4,data3);
    EXPECT_EQ(n4[0].as_uint32(),a_val);
    EXPECT_EQ(n4[1].as_float64(),c_val);
    EXPECT_EQ(n4[2].as_uint32(),b_val);

    Schema schema5("{top:[ {int1: uint32, int2: uint32}, float64, uint32], other: float64}");
    char* data4 = new char[28];
    uint32   d_val  = 40;
    float64  e_val  = 50.0;
    memcpy(&data4[0],&a_val,4);
    memcpy(&data4[4],&b_val,4);
    memcpy(&data4[8],&c_val,8);
    memcpy(&data4[16],&d_val,4);
    memcpy(&data4[20],&e_val,8);
    Node n5(schema5,data4);
    
    n5.schema().print();

    EXPECT_EQ(n5["top"][0]["int1"].as_uint32(),a_val);
    EXPECT_EQ(n5["top"][0]["int2"].as_uint32(),b_val);
    EXPECT_EQ(n5["top"][1].as_float64(),c_val);
    EXPECT_EQ(n5["top"][2].as_uint32(),d_val);
    EXPECT_EQ(n5["other"].as_float64(),e_val);

}

