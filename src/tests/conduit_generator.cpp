//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// file: conduit_generator.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;


TEST(conduit_generator_simple_gen_schema_test, conduit_generator)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);

    Generator g1("{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}",data);
    Node n;
    g1.walk(n);
    
    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);

    std::string s2_str = "{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}";
    std::cout << s2_str << std::endl;
    Generator g2(s2_str);
    Schema schema2;
    g2.walk(schema2);
    
    Node n2(schema2,data);
    EXPECT_EQ(n2["g"]["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["g"]["b"].as_uint32(),b_val);
    EXPECT_EQ(n2["g"]["c"].as_float64(),c_val);

    uint32 *data2 = new uint32[5];
    for (int i = 0; i < 5; i++) {
       data2[i] = i * 5;
    }

    Generator g3("{\"dtype\":\"uint32\",\"length\": 5}");
    Schema schema3;
    g3.walk(schema3);
    Node n3(schema3,data2);

    for (int i = 0; i < 5; i++) {
       EXPECT_EQ(n3.as_uint32_ptr()[i], i*5);
    }

    Generator g4("[\"uint32\", \"float64\", \"uint32\"]");

    char* data3 = new char[16];
    memcpy(&data3[0],&a_val,4);
    memcpy(&data3[4],&c_val,8);
    memcpy(&data3[12],&b_val,4);
    Schema schema4;
    g4.walk(schema4);
    Node n4(schema4,data3);
    EXPECT_EQ(n4[0].as_uint32(),a_val);
    EXPECT_EQ(n4[1].as_float64(),c_val);
    EXPECT_EQ(n4[2].as_uint32(),b_val);

    Generator g5("{\"top\":[{\"int1\":\"uint32\", \"int2\":\"uint32\"}, \"float64\", \"uint32\"], \"other\":\"float64\"}");
    Schema schema5;
    g5.walk(schema5);
        
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


