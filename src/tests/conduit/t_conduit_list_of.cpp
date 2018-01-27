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
/// file: conduit_list_of.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_list_of, simple )
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
    Node n(sch,data,true);
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

    delete [] data;
}

//-----------------------------------------------------------------------------
TEST(conduit_list_of, path_ref)
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
    Node n(jschema,data,true);
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

    delete [] data;
}
    
    

//-----------------------------------------------------------------------------
TEST(conduit_list_of, without_json)
{
    uint32  len_val = 2;
    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[16];
    memcpy(&data[0],&a1_val,4);    
    memcpy(&data[4],&b1_val,4);
    memcpy(&data[8],&a2_val,4);
    memcpy(&data[12],&b2_val,4);
    
    Node n;
    
    Schema entry_s;
    entry_s["a"].set(DataType::int32());
    entry_s["b"].set(DataType::int32());
    
    n["list_length"] = len_val;
    n["values"].list_of_external(data,entry_s,2);
    
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


    // test the non-external case:
    Node n2;
    n2.list_of(entry_s,2);
    n2.update(n["values"]);
    n2.print();
    
    // we should only have one allocation 
    Node ninfo;
    n2.info(ninfo);
    ninfo.print();
    

    // test if n2 was actually created in a contiguous way
    Node n3;
    n3.set_external((int32*)n2.data_ptr(),4);
    n3.print();
    
    int32 *n3_ptr = n3.value();
    
    EXPECT_EQ(n3_ptr[0], a1_val);
    EXPECT_EQ(n3_ptr[1], b1_val);
    EXPECT_EQ(n3_ptr[2], a2_val);
    EXPECT_EQ(n3_ptr[3], b2_val);

    delete [] data;
}
    
    

//-----------------------------------------------------------------------------
TEST(conduit_list_of, dtype)
{
    Node n;
    n.list_of(DataType::empty(),5);
    n.info().print();
    n.print();
    EXPECT_TRUE(n.dtype().is_list());
    EXPECT_EQ(n.number_of_children(),5);
    
    
    n.list_of(DataType::float64(2),5);
    n.print();
    n.info().print();
    EXPECT_EQ(n.allocated_bytes(), 5 * 2 * 8);
    

    NodeIterator itr = n.children();
    while(itr.has_next())
    {
        float64_array vals = itr.next().value();
        vals[0] = 10;
        vals[1] = 20;
    }    
    
    n.print();
    
    
}
