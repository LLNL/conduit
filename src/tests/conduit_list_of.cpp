///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;


TEST(conduit_list_of_simple, conduit_list_of)
{
    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[16];
    memcpy(&data[0],&a1_val,4);
    memcpy(&data[4],&b1_val,4);
    memcpy(&data[8],&a2_val,4);
    memcpy(&data[12],&b2_val,4);
    
    std::string jschema = "{\"dtype\":{\"a\":\"int32\",\"b\":\"int32\"},\"length\":2}";
	Schema sch(jschema);
    Node n(sch,data);
    std::cout << n.schema().to_json() << std::endl;
    std::cout << n.to_json() << std::endl;
    
    std::cout <<  n[0]["a"].as_int32() << std::endl;
    std::cout <<  n[1]["a"].as_int32() << std::endl;

    std::cout <<  n[0]["b"].as_int32() << std::endl;
    std::cout <<  n[1]["b"].as_int32() << std::endl;

    EXPECT_EQ(n[0]["a"].as_int32(), a1_val);
    EXPECT_EQ(n[1]["a"].as_int32(), a2_val);

    EXPECT_EQ(n[0]["b"].as_int32(), b1_val);
    EXPECT_EQ(n[1]["b"].as_int32(), b2_val);
}

TEST(conduit_list_of_path_ref, conduit_list_of)
{
    uint32  len_val = 2;
    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[20];
    memcpy(&data[0],&len_val,4);    
    memcpy(&data[4],&a1_val,4);
    memcpy(&data[8],&b1_val,4);
    memcpy(&data[12],&a2_val,4);
    memcpy(&data[16],&b2_val,4);
    
    std::string jschema = "{ \"list_length\": \"uint32\", \"values\":{\"dtype\":{\"a\":\"int32\",\"b\":\"int32\"},\"length\":{\"reference\":\"../list_length\"}}}";
	
    //For ref case, we will have to pass the json directly
    //
    Node n(jschema,data);
    std::cout << n.schema().to_json() << std::endl;
    std::cout << n.to_json() << std::endl;

    std::cout <<  n["list_length"].as_uint32() << std::endl;

    std::cout <<  n["values"][0]["a"].as_int32() << std::endl;
    std::cout <<  n["values"][1]["a"].as_int32() << std::endl;

    std::cout <<  n["values"][0]["b"].as_int32() << std::endl;
    std::cout <<  n["values"][1]["b"].as_int32() << std::endl;

    EXPECT_EQ(n["list_length"].as_uint32(), len_val);
    EXPECT_EQ(n["values"][0]["a"].as_int32(), a1_val);
    EXPECT_EQ(n["values"][1]["a"].as_int32(), a2_val);

    EXPECT_EQ(n["values"][0]["b"].as_int32(), b1_val);
    EXPECT_EQ(n["values"][1]["b"].as_int32(), b2_val);
}
    
