//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_generator.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_generator, simple_gen_schema)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);

    Generator g1("{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}",
                 "conduit_json",
                 data);
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
    
    Node n2(schema2,data,true); // true for external
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
    Node n3(schema3,data2,true); // true for external

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
    Node n4(schema4,data3,true); // true for external
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
    
    Node n5(schema5,data4,true); // true for external
    
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
TEST(conduit_generator, simple_gen_schema_with_gen_setters)
{
    uint32   a_val = 10;
    uint32   b_val = 20;
    float64  c_val = 30.0;

    char *data = new char[16];
    memcpy(&data[0], &a_val, 4);
    memcpy(&data[4], &b_val, 4);
    memcpy(&data[8], &c_val, 8);

    Generator g;

    std::string s1_str = "{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}";

    g.set_schema(s1_str);
    g.set_data_ptr(data);

    EXPECT_EQ(g.schema(), s1_str);
    EXPECT_EQ(g.protocol(), std::string("conduit_json"));
    EXPECT_EQ(g.data_ptr(), data);
    
    Node n;
    g.walk(n);

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_float64(), c_val);

    std::string s2_str = "{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}";
    std::cout << s2_str << std::endl;

    g.set_schema(s2_str);
    g.set_data_ptr(NULL);

    Schema schema2;
    g.walk(schema2);

    Node n2(schema2, data, true); // true for external
    EXPECT_EQ(n2["g"]["a"].as_uint32(), a_val);
    EXPECT_EQ(n2["g"]["b"].as_uint32(), b_val);
    EXPECT_EQ(n2["g"]["c"].as_float64(), c_val);

    uint32 *data2 = new uint32[5];
    for (int i = 0; i < 5; i++) {
        data2[i] = i * 5;
    }

    g.set_schema("{\"dtype\":\"uint32\",\"length\":  5}");

    Schema schema3;
    g.walk(schema3);
    Node n3(schema3, data2, true); // true for external

    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(n3.as_uint32_ptr()[i], i * 5);
    }

    g.set_schema("[\"uint32\", \"float64\", \"uint32\"]");

    char* data3 = new char[16];
    memcpy(&data3[0], &a_val, 4);
    memcpy(&data3[4], &c_val, 8);
    memcpy(&data3[12], &b_val, 4);
    Schema schema4;
    g.walk(schema4);
    Node n4(schema4, data3, true); // true for external
    EXPECT_EQ(n4[0].as_uint32(), a_val);
    EXPECT_EQ(n4[1].as_float64(), c_val);
    EXPECT_EQ(n4[2].as_uint32(), b_val);

    g.set_schema("{\"top\":[{\"int1\":\"uint32\", \"int2\":\"uint32\"}, \"float64\", \"uint32\"], \"other\":\"float64\"}");
    
    Schema schema5;
    g.walk(schema5);

    char* data4 = new char[28];
    uint32   d_val = 40;
    float64  e_val = 50.0;
    memcpy(&data4[0], &a_val, 4);
    memcpy(&data4[4], &b_val, 4);
    memcpy(&data4[8], &c_val, 8);
    memcpy(&data4[16], &d_val, 4);
    memcpy(&data4[20], &e_val, 8);

    Node n5(schema5, data4, true); // true for external

    n5.schema().print();
    EXPECT_EQ(n5["top"][0]["int1"].as_uint32(), a_val);
    EXPECT_EQ(n5["top"][0]["int2"].as_uint32(), b_val);
    EXPECT_EQ(n5["top"][1].as_float64(), c_val);
    EXPECT_EQ(n5["top"][2].as_uint32(), d_val);
    EXPECT_EQ(n5["other"].as_float64(), e_val);


    delete[] data;
    delete[] data2;
    delete[] data3;
    delete[] data4;

}


//-----------------------------------------------------------------------------
TEST(conduit_generator, gen_array_with_num_eles)
{
    Node n;

    // signed ints
    n.generate("{\"dtype\":\"int8\",\"number_of_elements\": 8}");
    EXPECT_TRUE(n.dtype().is_int8());
    EXPECT_EQ(8,n.dtype().number_of_elements());
    
    n.generate("{\"dtype\":\"int16\",\"number_of_elements\": 16}");
    EXPECT_TRUE(n.dtype().is_int16());
    EXPECT_EQ(16,n.dtype().number_of_elements());

    n.generate("{\"dtype\":\"int32\",\"number_of_elements\": 32}");
    EXPECT_TRUE(n.dtype().is_int32());
    EXPECT_EQ(32,n.dtype().number_of_elements());
    
    n.generate("{\"dtype\":\"int64\",\"number_of_elements\": 64}");
    EXPECT_TRUE(n.dtype().is_int64());
    EXPECT_EQ(64,n.dtype().number_of_elements());

    // unsigned ints
    n.generate("{\"dtype\":\"uint8\",\"number_of_elements\": 8}");
    EXPECT_TRUE(n.dtype().is_uint8());
    EXPECT_EQ(8,n.dtype().number_of_elements());
    
    n.generate("{\"dtype\":\"uint16\",\"number_of_elements\": 16}");
    EXPECT_TRUE(n.dtype().is_uint16());
    EXPECT_EQ(16,n.dtype().number_of_elements());

    n.generate("{\"dtype\":\"uint32\",\"number_of_elements\": 32}");
    EXPECT_TRUE(n.dtype().is_uint32());
    EXPECT_EQ(32,n.dtype().number_of_elements());
    
    n.generate("{\"dtype\":\"uint64\",\"number_of_elements\": 64}");
    EXPECT_TRUE(n.dtype().is_uint64());
    EXPECT_EQ(64,n.dtype().number_of_elements());
    
    n.generate("{\"dtype\":\"float32\",\"number_of_elements\": 32}");
    EXPECT_TRUE(n.dtype().is_float32());
    EXPECT_EQ(32,n.dtype().number_of_elements());
    
    n.generate("{\"dtype\":\"float64\",\"number_of_elements\": 64}");
    EXPECT_TRUE(n.dtype().is_float64());
    EXPECT_EQ(64,n.dtype().number_of_elements());
    
}

//-----------------------------------------------------------------------------
TEST(conduit_generator, gen_array_with_data)
{
    Node n;
    // signed ints
    n.generate("{\"dtype\":\"int8\",\"length\": 2, \"value\": [-8,-8]}");
    int8 *vint8_ptr = n.value();
    EXPECT_EQ(-8,vint8_ptr[0]);
    EXPECT_EQ(vint8_ptr[0],vint8_ptr[1]);

    n.generate("{\"dtype\":\"int16\",\"length\": 2, \"value\": [-16,-16]}");
    int16 *vint16_ptr = n.value();
    EXPECT_EQ(-16,vint16_ptr[0]);
    EXPECT_EQ(vint16_ptr[0],vint16_ptr[1]);

    n.generate("{\"dtype\":\"int32\",\"length\": 2, \"value\": [-32,-32]}");
    int32 *vint32_ptr = n.value();
    EXPECT_EQ(-32,vint32_ptr[0]);
    EXPECT_EQ(vint32_ptr[0],vint32_ptr[1]);

    n.generate("{\"dtype\":\"int64\",\"length\": 2, \"value\": [-64,-64]}");
    int64 *vint64_ptr = n.value();
    EXPECT_EQ(-64,vint64_ptr[0]);
    EXPECT_EQ(vint64_ptr[0],vint64_ptr[1]);
    
    // unsigned ints
    n.generate("{\"dtype\":\"uint8\",\"length\": 2, \"value\": [8,8]}");
    uint8 *vuint8_ptr = n.value();
    EXPECT_EQ(8,vuint8_ptr[0]);
    EXPECT_EQ(vuint8_ptr[0],vuint8_ptr[1]);

    n.generate("{\"dtype\":\"uint16\",\"length\": 2, \"value\": [16,16]}");
    uint16 *vuint16_ptr = n.value();
    EXPECT_EQ(16,vuint16_ptr[0]);
    EXPECT_EQ(vuint16_ptr[0],vuint16_ptr[1]);

    n.generate("{\"dtype\":\"uint32\",\"length\": 2, \"value\": [32,32]}");
    uint32 *vuint32_ptr = n.value();
    EXPECT_EQ(32,vuint32_ptr[0]);
    EXPECT_EQ(vuint32_ptr[0],vuint32_ptr[1]);

    n.generate("{\"dtype\":\"uint64\",\"length\": 2, \"value\": [64,64]}");
    uint64 *vuint64_ptr = n.value();
    EXPECT_EQ(64,vuint64_ptr[0]);
    EXPECT_EQ(vuint64_ptr[0],vuint64_ptr[1]);

    // floating point
    n.generate("{\"dtype\":\"float32\",\"length\": 2, \"value\": [32.0,32.0]}");
    float32 *vfloat32_ptr = n.value();
    EXPECT_NEAR(32,vfloat32_ptr[0],1e-10);
    EXPECT_EQ(vfloat32_ptr[0],vfloat32_ptr[1]);

    n.generate("{\"dtype\":\"float64\",\"length\": 2, \"value\": [64.0,64.0]}");
    float64 *vfloat64_ptr = n.value();
    EXPECT_NEAR(64,vfloat64_ptr[0],1e-10);
    EXPECT_EQ(vfloat64_ptr[0],vfloat64_ptr[1]);

    
    
    n.generate("{\"dtype\":\"char8_str\",\"value\": \"mystring\"}");
    
    EXPECT_EQ("mystring",n.as_string());

}

//-----------------------------------------------------------------------------
TEST(conduit_generator, gen_endianness)
{
    union{uint8  vbytes[4]; uint32 vuint;} data;

    if(Endianness::machine_default() == Endianness::BIG_ID)
    {
        data.vbytes[0] =  0xff;
        data.vbytes[1] =  0xff;
        data.vbytes[2] =  0xff;
        data.vbytes[3] =  0xfe;
        
      
        EXPECT_EQ(0xfffffffe,data.vuint);
        
        CONDUIT_INFO("Gen as Big Endian (Machine Default)");
        Generator g1("{\"dtype\":\"uint32\",\"length\": 1, \"endianness\": \"big\"}",
                     "conduit_json",
                     &data.vbytes[0]);
      
        Node n;
        n.generate_external(g1);
        
        EXPECT_EQ(0xfffffffe,n.as_uint32());
        
        
        data.vbytes[0] =  0xfe;
        data.vbytes[1] =  0xff;
        data.vbytes[2] =  0xff;
        data.vbytes[3] =  0xff;
        
        CONDUIT_INFO("Gen as Little Endian");
        Generator g2("{\"dtype\":\"uint32\",\"length\": 1, \"endianness\": \"little\"}",
                     "conduit_json",
                     &data.vbytes[0]);
        
        n.generate_external(g2);
        n.endian_swap_to_machine_default();
        EXPECT_EQ(0xfffffffe,n.as_uint32());
        
    }
    else
    {
        data.vbytes[0] =  0xfe;
        data.vbytes[1] =  0xff;
        data.vbytes[2] =  0xff;
        data.vbytes[3] =  0xff;
      
        EXPECT_EQ(0xfffffffe,data.vuint);
        
        CONDUIT_INFO("Gen as Little Endian (Machine Default)");
        Generator g("{\"dtype\":\"uint32\",\"length\": 1, \"endianness\": \"little\"}",
                    "conduit_json",
                    &data.vbytes[0]);
      
        Node n;
        n.generate_external(g);
        n.print_detailed();
                
        EXPECT_EQ(0xfffffffe,n.as_uint32());

        data.vbytes[0] =  0xff;
        data.vbytes[1] =  0xff;
        data.vbytes[2] =  0xff;
        data.vbytes[3] =  0xfe;
    
        Generator g2("{\"dtype\":\"uint32\",\"length\": 1, \"endianness\": \"big\"}",
                     "conduit_json",
                     &data.vbytes[0]);
        
        CONDUIT_INFO("Gen as Big Endian");
        n.generate_external(g2);
        n.print_detailed();
        n.endian_swap_to_machine_default();
        EXPECT_EQ(0xfffffffe,n.as_uint32());
        
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_generator, simple_gen_schema_yaml)
{
    Generator g1("a: 10\nb: 20\nc: \"30\"\nd:\n  - hi\n  - there",
                 "yaml");
    Node n;
    g1.walk(n);
    n.print();

    EXPECT_EQ(n["a"].as_int64(),10);
    EXPECT_EQ(n["b"].as_int64(),20);
    EXPECT_EQ(n["c"].as_int64(),30);
    EXPECT_EQ(n["d"][0].as_string(),"hi");
    EXPECT_EQ(n["d"][1].as_string(),"there");
    
    Generator g2("a: 10\nb: 20\nc: \"30\"\nd: [0, 10, 20, 30]\n",
                 "yaml");
    g2.walk(n);
    n.print();

    EXPECT_EQ(n["a"].as_int64(),10);
    EXPECT_EQ(n["b"].as_int64(),20);
    EXPECT_EQ(n["c"].as_int64(),30);
    int64_array d_vals = n["d"].value();
    EXPECT_EQ(d_vals[0],0);
    EXPECT_EQ(d_vals[1],10);
    EXPECT_EQ(d_vals[2],20);
    EXPECT_EQ(d_vals[3],30);
    
    
    // these are no longer special cases
    // we handle as string, but keep checking them
    
    Generator g3("a: true\nb: false\nc: null\n",
                 "yaml");
    g3.walk(n);
    n.print();

    EXPECT_TRUE(n["a"].dtype().is_string());
    EXPECT_TRUE(n["b"].dtype().is_string());
    EXPECT_TRUE(n["c"].dtype().is_string());

    EXPECT_EQ(n["a"].as_string(),"true");
    EXPECT_EQ(n["b"].as_string(),"false");
    EXPECT_EQ(n["c"].as_string(),"null");

}



//-----------------------------------------------------------------------------
TEST(conduit_generator, yaml_parsing_errors)
{
    Generator g("a: 10\ns","yaml");
    Node n;
    EXPECT_THROW(g.walk(n),conduit::Error);

    // protocol will still be "yaml"
    g.set_schema("[ 10,\ns");
    EXPECT_THROW(g.walk(n),conduit::Error);
}


