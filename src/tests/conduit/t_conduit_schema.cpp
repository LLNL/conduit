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
// For details, see: http://llnl.github.io/conduit/.
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
/// file: t_conduit_schema.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"


using namespace conduit;


//-----------------------------------------------------------------------------
TEST(schema_basics, construction)
{
    Schema s1;
    // const from dtype
    Schema s_b(DataType::float64(20));
    
    s1["a"].set(DataType::int64(10));
    s1["b"] = s_b;
    
    // copy const
    Schema s2(s1);
    
    EXPECT_TRUE(s1.equals(s2));
    
    EXPECT_EQ(s2[1].parent(),&s2);
    
    EXPECT_EQ(s2.fetch_child("a").dtype().id(),DataType::INT64_ID);
    
}


//-----------------------------------------------------------------------------
TEST(schema_basics, equal_schemas)
{
    Schema s1;
    s1["a"].set(DataType::int64(10));
    s1["b"].set(DataType::float64(20));
    
    
    
    Schema s2;
    s2["a"].set(DataType::int64(10));
    s2["b"].set(DataType::float64(20));
    
    EXPECT_TRUE(s1.equals(s2));
    
    
    s1["c"].set(DataType::uint8(20));

    EXPECT_FALSE(s1.equals(s2));
}


//-----------------------------------------------------------------------------
TEST(schema_basics, compatible_schemas)
{
    Schema s1;
    s1["a"].set(DataType::int64(10));
    s1["b"].set(DataType::float64(20,s1["a"].total_bytes()));
    
    std::string s2_json  = "{ a: {dtype:int64, length:10 }, ";
    s2_json              += " b: {dtype:float64, length:20 } }";
    
    Schema s2;
    s2 = s2_json;
    EXPECT_TRUE(s1.compatible(s2));
    EXPECT_TRUE(s1.equals(s2));

    //
    // a.compat(b) means:
    //  "all the entries of b, can be copied into a without any new allocs"
    // 
    // in this case, s1.compat(s3) is not ok, but the reverse is
    std::string s3_json  = "{ a: {dtype:int64, length:10 }, ";
    s3_json              += " b: {dtype:float64, length:40} }";
    
    
    Schema s3(s3_json);
    EXPECT_FALSE(s1.compatible(s3));
    EXPECT_TRUE(s3.compatible(s1));
}


//-----------------------------------------------------------------------------
TEST(schema_basics, compatible_schemas_with_lists)
{
    Schema s1;
    Schema &s1_a = s1.append();
    Schema &s1_b = s1.append();
    
    s1_a.set(DataType::int8(10));
    s1_b.set(DataType::int8(10));

    Schema s2;
    Schema &s2_a = s2.append();
    Schema &s2_b = s2.append();
    Schema &s2_c = s2.append();

    s2_a.set(DataType::int8(10));
    s2_b.set(DataType::int8(10));
    s2_c.set(DataType::int8(10));
    
    EXPECT_FALSE(s1.compatible(s2));
    EXPECT_TRUE(s2.compatible(s1));

    EXPECT_FALSE(s1.equals(s2));
    EXPECT_TRUE(s1.compatible(s1));


}

//-----------------------------------------------------------------------------
TEST(schema_basics, schema_alloc)
{
    Schema s1;
    s1["a"].set(DataType::int64(10));
    s1["b"].set(DataType::float64(20,s1.total_bytes()));
    // pad
    s1["c"].set(DataType::float64(1,s1.total_bytes()+ 10));
    
    EXPECT_EQ(s1.total_bytes(), sizeof(int64) * 10  + sizeof(float64) * 21);
    
    Node n1(s1);

    // this is what we need & this does work
    EXPECT_EQ(n1.allocated_bytes(),
              sizeof(int64) * 10  + sizeof(float64) * 21 + 10);
    
    // but given this, the following is strange:
    EXPECT_EQ(n1.total_bytes(), s1.total_bytes());
    // total_bytes doesn't seem like a good name

}

//-----------------------------------------------------------------------------
TEST(schema_basics, schema_name_by_index)
{
    Schema s1;
    s1["a"].set(DataType::int64());
    s1["b"].set(DataType::float64());
    s1["c"].set(DataType::float64());
    
    // standard case
    EXPECT_EQ(s1.child_name(0),"a");
    EXPECT_EQ(s1.child_name(1),"b");
    EXPECT_EQ(s1.child_name(2),"c");

    // these are out of bounds, should be empty
    EXPECT_EQ(s1.child_name(100),"");
    EXPECT_EQ(s1["a"].child_name(100),"");
    
    Schema s2;
    // check empty schema
    EXPECT_EQ(s2.child_name(100),"");
}

//-----------------------------------------------------------------------------
TEST(schema_basics, schema_fetch_child)
{
    Schema s;
    s["a"].set(DataType::int64());
    s["b"].set(DataType::float64());
    s["c"].set(DataType::float64());

    const Schema &s_c = s["c"];

    EXPECT_THROW(s.fetch_child("bad"),conduit::Error);
    EXPECT_THROW(const Schema &s_bad = s_c.fetch_child("bad"),conduit::Error);
}


//-----------------------------------------------------------------------------
///
/// commented out b/c spanned_bytes is now private, 
/// keeping if useful in future
/// 
//-----------------------------------------------------------------------------
// TEST(schema_basics, total_vs_spanned_bytes)
// {
//     Schema s;
//
//     s.set(DataType::int64(10));
//
//     EXPECT_EQ(s.total_bytes(),sizeof(int64) * 10);
//
//     s["a"].set(DataType::int64(10));
//     s["b"].set(DataType::int64(10,80));
//     s["c"].set(DataType::int64(10,160));
//
//
//     EXPECT_EQ(s.total_bytes(),8 * 10 * 3);
//     EXPECT_EQ(s.spanned_bytes(),s.total_bytes());
//
//     // at this point, we have a compact layout
//     EXPECT_TRUE(s.is_compact());
//
//     // add a new child, with an offset further than the last array len
//     s["d"].set(DataType::int64(10,320));
//
//     // now our spanned bytes is wider than total_bytes
//     EXPECT_EQ(s.spanned_bytes(),400);
//     EXPECT_EQ(s.total_bytes(),8 * 10 * 4);
//     EXPECT_LT(s.total_bytes(),s.spanned_bytes());
// }


