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
    
    EXPECT_EQ(s2.fetch_existing("a").dtype().id(),DataType::INT64_ID);
    
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
    
    Schema s3(DataType::float64());
    EXPECT_TRUE(s3.equals(s3));

    EXPECT_FALSE(s1.equals(s3));


    Schema s4;
    s4.append().set(DataType::float64());
    s4.append().set(DataType::float64());
    s4.append().set(DataType::float64());

    Schema s5(s4);
    
    EXPECT_TRUE(s4.equals(s5));

    s5.append().set(DataType::float64());
    EXPECT_FALSE(s4.equals(s5));

    s5.remove(3);

    EXPECT_TRUE(s4.equals(s5));
}


//-----------------------------------------------------------------------------
TEST(schema_basics, compatible_schemas)
{
    Schema s1;
    s1["a"].set(DataType::int64(10));
    s1["b"].set(DataType::float64(20,s1["a"].total_strided_bytes()));
    
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
    s1["b"].set(DataType::float64(20,s1.total_strided_bytes()));
    // pad
    s1["c"].set(DataType::float64(1,s1.total_strided_bytes()+ 10));
    
    EXPECT_EQ(s1.total_strided_bytes(), sizeof(int64) * 10  + sizeof(float64) * 21);
    
    Node n1(s1);

    // this is what we need & this does work
    EXPECT_EQ(n1.allocated_bytes(),
              sizeof(int64) * 10  + sizeof(float64) * 21 + 10);

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
TEST(schema_basics, schema_fetch_existing)
{
    Schema s;
    s["a"].set(DataType::int64());
    s["b"].set(DataType::float64());
    s["c"].set(DataType::float64());
    s["d/e"].set(DataType::int64());

    const Schema &s_c = s["c"];

    const Schema &s_c_idx = s[2];
    
    EXPECT_EQ(&s_c,&s_c_idx);

    EXPECT_THROW(s.fetch_existing("bad"),conduit::Error);
    EXPECT_THROW(const Schema &s_bad = s_c.fetch_existing("bad"),conduit::Error);
    
    const Schema *s_d_ptr = s.fetch_ptr("d");

    EXPECT_TRUE(s_d_ptr->dtype().is_object());
    
    const Schema &s_e = s_d_ptr->fetch("e");

    EXPECT_TRUE(s_e.dtype().is_int64());

}



//-----------------------------------------------------------------------------
TEST(schema_basics, schema_child_names)
{
    Schema s;
    s["a"].set(DataType::int64());
    s["b"].set(DataType::float64());
    s["c"].set(DataType::float64());
    s["d/e"].set(DataType::int64());
    
    const std::vector<std::string> &cld_names = s.child_names();
    
    const std::vector<std::string> &sde_names = s["d/e"].child_names();

    EXPECT_EQ(cld_names[0],std::string("a"));
    EXPECT_EQ(cld_names[1],std::string("b"));
    EXPECT_EQ(cld_names[2],std::string("c"));
    EXPECT_EQ(cld_names[3],std::string("d"));

    EXPECT_EQ(sde_names.size(),0);

}


//-----------------------------------------------------------------------------
TEST(schema_basics, schema_errors)
{
    Schema s;
    s["a"].set(DataType::int64());
    s["b"].set(DataType::float64());
    s["c"].set(DataType::float64());
    
    EXPECT_THROW(s.save("/dev/null/bad"),conduit::Error);
    EXPECT_THROW(s.load("/dev/null/bad"),conduit::Error);

    s.reset(); // can remove from empty
    EXPECT_THROW(s.remove(0),conduit::Error);
    s.append(); // now we have a list with one element
    EXPECT_THROW(s.remove(1),conduit::Error);
    
    s.reset();
    EXPECT_THROW(s.remove("a"),conduit::Error);
    EXPECT_THROW(s.fetch_existing("a"),conduit::Error);
    s = DataType::object();
    EXPECT_THROW(s.fetch_existing(".."),conduit::Error);
}

//-----------------------------------------------------------------------------
TEST(schema_basics, rename_child)
{
    Schema s;
    
    // error, can't rename non object;
    EXPECT_THROW(s.rename_child("a","b"),conduit::Error);
    
    s["a"].set(DataType::int64());
    s["b"].set(DataType::float64());
    s["c"].set(DataType::float32(10));
    
    s.print();

    // error, can't rename to existing child name
    EXPECT_THROW(s.rename_child("a","b"),conduit::Error);

    // error, can't rename non existing child
    EXPECT_THROW(s.rename_child("bad","d"),conduit::Error);

    std::vector<std::string> cnames = s.child_names();
    EXPECT_EQ(cnames[2],"c");
    EXPECT_TRUE(s.has_child("c"));
    EXPECT_FALSE(s.has_child("d"));

    s.rename_child("c","d");

    cnames = s.child_names();
    EXPECT_TRUE(s.has_child("d"));
    EXPECT_FALSE(s.has_child("c"));
    EXPECT_EQ(cnames[2],"d");
}


//-----------------------------------------------------------------------------
TEST(schema_basics, pathlike_child_names)
{
    Schema s;
    
    std::string shared_name = "a/b";
    std::string path_only = "c/d";
    std::string direct_only = "e/f";

    s[shared_name].set(DataType::int64());
    s.add_child(shared_name).set(DataType::int64());

    s[path_only].set(DataType::int64());
    s.add_child(direct_only).set(DataType::int64());
    
    s.print();

    EXPECT_TRUE(s.has_child(shared_name));
    EXPECT_TRUE(s.has_path(shared_name));

    EXPECT_TRUE(s.has_path(path_only));
    EXPECT_FALSE(s.has_path(direct_only));

    EXPECT_TRUE(s.has_child(direct_only));
    EXPECT_FALSE(s.has_child(path_only));

    // Test that explicitly removing children doesn't remove
    // by path and vice-versa 
    std::string second_shared_name = "foo/bar";
    s[second_shared_name].set(DataType::int64());
    s.add_child(second_shared_name).set(DataType::int64());

    s.remove(shared_name);
    s.remove_child(second_shared_name);

    EXPECT_TRUE(s.has_child(shared_name));
    EXPECT_FALSE(s.has_path(shared_name));

    EXPECT_TRUE(s.has_path(second_shared_name));
    EXPECT_FALSE(s.has_child(second_shared_name));

    s.print();

    // check compact_to , equal and compatible
    Schema s2;
    s["a"].set(DataType::int64());
    s.add_child("key_with_/_ex").set(DataType::int64());
    
    EXPECT_EQ(s.child("key_with_/_ex").path(),"{key_with_/_ex}");
    
    Schema s3(s2);
    
    EXPECT_TRUE(s2.equals(s3));
    EXPECT_TRUE(s2.compatible(s3));
    
    Schema s2_compact;
    s2.compact_to(s2_compact);
    EXPECT_TRUE(s2_compact.is_compact());
    EXPECT_TRUE(s2.compatible(s2_compact));

}

//-----------------------------------------------------------------------------
TEST(schema_basics, schema_to_string)
{
    Schema s;
    s["a"].set(DataType::int64());
    s["b"].set(DataType::float64());
    s["c"].set(DataType::float64());

    std::string res_str  = s.to_string();
    std::string res_json = s.to_json();

    std::cout << res_str << std::endl;
    std::cout << res_json << std::endl;

    // we expect these to be the same
    EXPECT_EQ(res_str, res_json);
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
//     EXPECT_EQ(s.total_strided_bytes(),sizeof(int64) * 10);
//
//     s["a"].set(DataType::int64(10));
//     s["b"].set(DataType::int64(10,80));
//     s["c"].set(DataType::int64(10,160));
//
//
//     EXPECT_EQ(s.total_strided_bytes(),8 * 10 * 3);
//     EXPECT_EQ(s.spanned_bytes(),s.total_strided_bytes());
//
//     // at this point, we have a compact layout
//     EXPECT_TRUE(s.is_compact());
//
//     // add a new child, with an offset further than the last array len
//     s["d"].set(DataType::int64(10,320));
//
//     // now our spanned bytes is wider than total_bytes
//     EXPECT_EQ(s.spanned_bytes(),400);
//     EXPECT_EQ(s.total_strided_bytes(),8 * 10 * 4);
//     EXPECT_LT(s.total_strided_bytes(),s.spanned_bytes());
// }


