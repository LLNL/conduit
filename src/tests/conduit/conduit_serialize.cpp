//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
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
/// file: conduit_serialize.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;



//-----------------------------------------------------------------------------
TEST(conduit_serialize, test_1)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);


    n.schema().print();
    std::vector<uint8> bytes;

    Schema s_schema;
    n.serialize(bytes);
    n.schema().compact_to(s_schema);
    
    s_schema.print();

    std::cout << *((uint32*)&bytes[0]) << std::endl;

	Node n2(s_schema,&bytes[0],true);
    EXPECT_EQ(n2["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["b"].as_uint32(),b_val);
}

//-----------------------------------------------------------------------------
TEST(conduit_serialize, test_2)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    n["here"]["there"] = a_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);
    EXPECT_EQ(n["here"]["there"].as_uint32(),a_val);

    n.schema().print();
    std::vector<uint8> bytes;
    n.serialize(bytes);

    Schema c_schema;
    n.schema().compact_to(c_schema);
    Node n2(c_schema,&bytes[0],true);
    n2.schema().print();
    EXPECT_EQ(n2["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["b"].as_uint32(),b_val);
    EXPECT_EQ(n2["c"].as_float64(),c_val);
    EXPECT_EQ(n2["here"]["there"].as_uint32(),a_val);
    EXPECT_EQ(1,n2["here"].number_of_children());

}

//-----------------------------------------------------------------------------
TEST(conduit_serialize, compact)
{
    float64 vals[] = { 100.0,
                      -100.0, 
                       200.0,
                      -200.0,
                       300.0,
                      -300.0,
                       400.0,
                      -400.0,
                       500.0,
                      -500.0};

    Generator g("{dtype: float64, length: 5, stride: 16, offset:8}",vals);

    Node n(g,true);


    EXPECT_EQ(n.info()["total_bytes"].to_uint64(),80);
    
    std::vector<uint8> nc_bytes;
    Schema nc_s;
    
    n.schema().print();
    n.schema().compact_to(nc_s);
    nc_s.print();

    n.serialize(nc_bytes);
    EXPECT_EQ(nc_bytes.size(),40);
    
    Node nc(nc_s,&nc_bytes[0],true);


    EXPECT_EQ(n.as_float64_array()[1],-200.0);
    EXPECT_EQ(nc.as_float64_ptr()[1],-200.0);

    nc.schema().print();
    nc.print();
    nc.info().print();
    
    EXPECT_EQ(nc.info()["total_bytes"].to_uint64(),40);
    
}



