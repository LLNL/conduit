//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: conduit_node.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node, simple)
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

TEST(conduit_node, nested)
{

    uint32   val  = 10;

    Node n;
    n["a"]["b"] = val;
    EXPECT_EQ(n["a"]["b"].as_uint32(),val);
}

//-----------------------------------------------------------------------------
TEST(conduit_node, vector)
{

    std::vector<uint32> vec;
    for(int i=0;i<100;i++)
        vec.push_back(i);

    Node n;
    n["a"]= vec;
    EXPECT_EQ(n["a"].as_uint32_ptr()[99],99);
}

//-----------------------------------------------------------------------------
TEST(conduit_node, list)
{

    std::vector<uint32> vec;
    for(int i=0;i<100;i++)
        vec.push_back(i);

    Node n;
    Node& list = n["mylist"];
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;
    list.append().set(a_val);
    list.append().set(b_val);
    list.append().set(c_val);
    list.append().set(vec);
    EXPECT_EQ(list[0].as_uint32(),a_val);
    EXPECT_EQ(list[1].as_uint32(),b_val);
    EXPECT_EQ(list[2].as_float64(),c_val);
    EXPECT_EQ(list[3].as_uint32_ptr()[99],99);

    EXPECT_EQ(n["mylist"][0].as_uint32(),a_val);
    EXPECT_EQ(n["mylist"][1].as_uint32(),b_val);
    EXPECT_EQ(n["mylist"][2].as_float64(),c_val);
    EXPECT_EQ(n["mylist"][3].as_uint32_ptr()[99],99);

}

//-----------------------------------------------------------------------------
TEST(conduit_node, simple_schema_gen )
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);

    Schema schema("{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}");
    Node n(schema,data,true);

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);

    std::string s2_str = "{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}";
    std::cout << s2_str << std::endl;
    Schema schema2(s2_str);

    Node n2(schema2,data,true);
    EXPECT_EQ(n2["g"]["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["g"]["b"].as_uint32(),b_val);
    EXPECT_EQ(n2["g"]["c"].as_float64(),c_val);

    Schema schema3("{\"dtype\":\"uint32\",\"length\": 5}");
    uint32 *data2 = new uint32[5];
    for (int i = 0; i < 5; i++) {
       data2[i] = i * 5;
    }
    Node n3(schema3,data2,true);
    for (int i = 0; i < 5; i++) {
       EXPECT_EQ(n3.as_uint32_ptr()[i], i*5);
    }
    Schema schema4("[\"uint32\", \"float64\", \"uint32\"]");
    char* data3 = new char[16];
    memcpy(&data3[0],&a_val,4);
    memcpy(&data3[4],&c_val,8);
    memcpy(&data3[12],&b_val,4);
    Node n4(schema4,data3,true);
    EXPECT_EQ(n4[0].as_uint32(),a_val);
    EXPECT_EQ(n4[1].as_float64(),c_val);
    EXPECT_EQ(n4[2].as_uint32(),b_val);

    Schema schema5("{\"top\":[{\"int1\":\"uint32\", \"int2\":\"uint32\"}, \"float64\", \"uint32\"], \"other\":\"float64\"}");
    char* data4 = new char[28];
    uint32   d_val  = 40;
    float64  e_val  = 50.0;
    memcpy(&data4[0],&a_val,4);
    memcpy(&data4[4],&b_val,4);
    memcpy(&data4[8],&c_val,8);
    memcpy(&data4[16],&d_val,4);
    memcpy(&data4[20],&e_val,8);
    Node n5(schema5,data4,true);

    std::cout << n5.schema().to_json() << std::endl;
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
TEST(conduit_node, simple_schema)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    n["here"]["there"] = c_val;

    std::string res = n.schema().to_json();
    std::cout << res;
    rapidjson::Document d;
    d.Parse<0>(res.c_str());

    EXPECT_TRUE(d.HasMember("a"));
    EXPECT_TRUE(d.HasMember("b"));
    EXPECT_TRUE(d.HasMember("c"));
}

//-----------------------------------------------------------------------------
TEST(conduit_node, simple_schema_parent)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    n["here"]["there"] = c_val;

    EXPECT_FALSE(n.schema().has_parent());
    Node & na = n["a"];
    const Schema &na_schema =na.schema();
    EXPECT_TRUE(na_schema.has_parent());

}


//-----------------------------------------------------------------------------
TEST(conduit_node, in_place)
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

    Schema schema("{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}");
    Node n(schema,data,true);
    n["a"] = b_val;
    n["b"] = a_val;
    n["c"] = d_val;

    EXPECT_EQ(n["a"].as_uint32(), b_val);
    EXPECT_EQ(n["b"].as_uint32(), a_val);
    EXPECT_EQ(n["c"].as_float64(), d_val);

    EXPECT_EQ(*(uint32*)(&data[0]), b_val);
    EXPECT_EQ(*(uint32*)(&data[4]), a_val);
    EXPECT_EQ(*(float64*)(&data[8]), d_val);

    delete [] data;
}

//-----------------------------------------------------------------------------
TEST(conduit_node, remove_by_name)
{
    conduit::Generator g("{a:1,b:2,c:3}", "json");
    conduit::Node n(g,true);
    n.print();
    EXPECT_TRUE(n.has_path("a"));
    EXPECT_TRUE(n.has_path("b"));
    EXPECT_TRUE(n.has_path("c"));
    n.remove("a");
    n.print();
    EXPECT_FALSE(n.has_path("a"));
    EXPECT_TRUE(n.has_path("b"));
    EXPECT_TRUE(n.has_path("c"));
    n.remove("c");
    n.print();
    EXPECT_FALSE(n.has_path("a"));
    EXPECT_TRUE(n.has_path("b"));
    EXPECT_FALSE(n.has_path("c"));
    n.remove("b");
    n.print();
    EXPECT_FALSE(n.has_path("a"));
    EXPECT_FALSE(n.has_path("b"));
    EXPECT_FALSE(n.has_path("c"));
}

//-----------------------------------------------------------------------------
TEST(conduit_node, remove_by_index)
{
    conduit::Generator g("{a:1,b:2,c:3}", "json");
    conduit::Node n(g,true);
    n.print();
    EXPECT_TRUE(n.has_path("a"));
    EXPECT_TRUE(n.has_path("b"));
    EXPECT_TRUE(n.has_path("c"));
    n.remove(0);
    n.print();
    EXPECT_FALSE(n.has_path("a"));
    EXPECT_TRUE(n.has_path("b"));
    EXPECT_TRUE(n.has_path("c"));
    n.remove(1);
    n.print();
    EXPECT_FALSE(n.has_path("a"));
    EXPECT_TRUE(n.has_path("b"));
    EXPECT_FALSE(n.has_path("c"));
    n.remove(0);
    n.print();
    EXPECT_FALSE(n.has_path("a"));
    EXPECT_FALSE(n.has_path("b"));
    EXPECT_FALSE(n.has_path("c"));
    
    conduit::Generator g2("[{dtype:int64, value: 10},{dtype:int64, value: 20},{dtype:int64, value: 30}]");
    conduit::Node n2(g2,true);
    n2.print();
    n2.remove(1);
    n2.print();
    EXPECT_EQ(n2[0].to_uint64(), 10);
    EXPECT_EQ(n2[1].to_uint64(), 30);
    n2.remove(0);
    n2.print();    
    EXPECT_EQ(n2[0].to_uint64(), 30);
}

//-----------------------------------------------------------------------------
TEST(conduit_node, check_leaf_assert)
{

    conduit::Node n;
    int16 v = 64;
    n["v"] = v;
    n.print();
    EXPECT_THROW(n["v"].as_int8(),conduit::Error);
}

//-----------------------------------------------------------------------------
TEST(conduit_node, check_value_implict_c_type_cast)
{
    conduit::Node n;
    
    char  cv = -1;
    short sv = -2;
    int   iv = -3;
    long  lv = -4;
     
    unsigned char  ucv = 1;
    unsigned short usv = 2;
    unsigned int   uiv = 3;
    unsigned long  ulv = 4;
    
    float  fv = 1.2;
    double dv = 2.4;
    
    n["cv"] = cv;
    n["sv"] = sv;
    n["iv"] = iv;
    n["lv"] = lv;
    
    
    n["ucv"] = ucv;
    n["usv"] = usv;
    n["uiv"] = uiv;
    n["ulv"] = ulv;
    
    n["fv"] = fv;
    n["dv"] = dv;
    
     
    n.print();
    
    char  cv_r = n["cv"].value();
    short sv_r = n["sv"].value();
    int   iv_r = n["iv"].value();
    long  lv_r = n["lv"].value();
    
    
    EXPECT_EQ(cv,cv_r);
    EXPECT_EQ(sv,sv_r);
    EXPECT_EQ(iv,iv_r);
    EXPECT_EQ(lv,lv_r);

    unsigned char  ucv_r = n["ucv"].value();
    unsigned short usv_r = n["usv"].value();
    unsigned int   uiv_r = n["uiv"].value();
    unsigned long  ulv_r = n["ulv"].value();

    EXPECT_EQ(ucv,ucv_r);
    EXPECT_EQ(usv,usv_r);
    EXPECT_EQ(uiv,uiv_r);
    EXPECT_EQ(ulv,ulv_r);


    float fv_r = n["fv"].value();
    float dv_r = n["dv"].value();

    EXPECT_NEAR(fv,fv_r,0.001);
    EXPECT_NEAR(dv,dv_r,0.001);
}


//-----------------------------------------------------------------------------
TEST(conduit_node, check_value_implict_bitwidth_type_cast)
{
    conduit::Node n;
    
    int8  i8v  = -1;
    int16 i16v = -2;
    int32 i32v = -3;
    int64 i64v = -4;
     
    uint8  ui8v  = 1;
    uint16 ui16v = 2;
    uint32 ui32v = 3;
    uint64 ui64v = 4;

    float32 f32v = 1.2;
    float64 f64v = 2.4;
    
    n["i8v"]  = i8v;
    n["i16v"] = i16v;
    n["i32v"] = i32v;
    n["i64v"] = i64v;
    
    n["ui8v"]  = ui8v;
    n["ui16v"] = ui16v;
    n["ui32v"] = ui32v;
    n["ui64v"] = ui64v;
    
    n["f32v"] = f32v;
    n["f64v"] = f64v;
    
     
    n.print();
    
    int8  i8v_r  = n["i8v"].value();
    int16 i16v_r = n["i16v"].value();
    int32 i32v_r = n["i32v"].value();
    int64 i64v_r = n["i64v"].value();
    
    EXPECT_EQ(i8v,i8v_r);
    EXPECT_EQ(i16v,i16v_r);
    EXPECT_EQ(i32v,i32v_r);
    EXPECT_EQ(i64v,i64v_r);

    uint8  ui8v_r  = n["ui8v"].value();
    uint16 ui16v_r = n["ui16v"].value();
    uint32 ui32v_r = n["ui32v"].value();
    uint64 ui64v_r = n["ui64v"].value();
    
    EXPECT_EQ(ui8v,ui8v_r);
    EXPECT_EQ(ui16v,ui16v_r);
    EXPECT_EQ(ui32v,ui32v_r);
    EXPECT_EQ(ui64v,ui64v_r);


    float f32v_r = n["f32v"].value();
    float f64v_r = n["f64v"].value();

    EXPECT_NEAR(f32v,f32v_r,0.001);
    EXPECT_NEAR(f64v,f64v_r,0.001);
}



