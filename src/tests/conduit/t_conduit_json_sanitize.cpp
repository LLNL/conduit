//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_conduit_json_sanitize.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_json_sanitize, sanitize_comments)
{
    std::string t1_in  = "//comment\n{\"a\": \"uint32\", \"b\": \"uint32\" , \"c\": \"float64\"}\n// comment!";
    std::string t1_out = "{\"a\": \"uint32\", \"b\": \"uint32\" , \"c\": \"float64\"}\n";
    
    EXPECT_EQ(utils::json_sanitize(t1_in),t1_out);
}

//-----------------------------------------------------------------------------
TEST(conduit_json_sanitize, sanitize_quoteless)
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

//-----------------------------------------------------------------------------
TEST(conduit_json_sanitize, simple_quoteless_schema)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);

    Schema schema("{a: uint32, b: uint32 , c: float64}");
    Node n(schema,data,true);
    
    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);

    std::string s2_str("{g: {a: uint32, b: uint32 , c: float64}}");
    std::cout << s2_str << std::endl;
    Schema schema2(s2_str);
    
    Node n2(schema2,data,true);
    EXPECT_EQ(n2["g"]["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["g"]["b"].as_uint32(),b_val);
    EXPECT_EQ(n2["g"]["c"].as_float64(),c_val);
    
    Schema schema3("{dtype: uint32, length: 5}");
    uint32 *data2 = new uint32[5];
    for (int i = 0; i < 5; i++) {
       data2[i] = i * 5;
    }
    Node n3(schema3,data2,true);
    for (int i = 0; i < 5; i++) {
       EXPECT_EQ(n3.as_uint32_ptr()[i], i*5);
    }
    Schema schema4("[ uint32, float64, uint32]");
    char* data3 = new char[16];
    memcpy(&data3[0],&a_val,4);
    memcpy(&data3[4],&c_val,8);
    memcpy(&data3[12],&b_val,4);
    Node n4(schema4,data3,true);
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
    Node n5(schema5,data4,true);
    
    n5.schema().print();

    EXPECT_EQ(n5["top"][0]["int1"].as_uint32(),a_val);
    EXPECT_EQ(n5["top"][0]["int2"].as_uint32(),b_val);
    EXPECT_EQ(n5["top"][1].as_float64(),c_val);
    EXPECT_EQ(n5["top"][2].as_uint32(),d_val);
    EXPECT_EQ(n5["other"].as_float64(),e_val);

    delete [] data;
    delete [] data2;
    delete [] data3;
    delete [] data4;
}



//-----------------------------------------------------------------------------
TEST(conduit_json_sanitize, sci_notation)
{
    std::string json_in = "{ \"data\": { \"value\": 1.2e-11 } }";
    
    EXPECT_EQ(json_in,utils::json_sanitize(json_in));
    
}
