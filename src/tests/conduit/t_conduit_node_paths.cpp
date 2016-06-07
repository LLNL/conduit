//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
/// file: conduit_node_paths.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <vector>
#include <set>
#include <string>
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_paths, simple_path)
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
    std::cout << n.schema().to_json() << std::endl; 

    EXPECT_TRUE(n.has_path("a"));
    EXPECT_EQ(n.fetch("a").as_uint32(),a_val);

    EXPECT_TRUE(n.has_path("b"));
    EXPECT_EQ(n.fetch("b").as_uint32(),b_val);

    EXPECT_TRUE(n.has_path("c"));
    EXPECT_EQ(n.fetch("c").as_float64(),c_val);

    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data,true);
    std::cout << n2.schema().to_json() << std::endl; 
    EXPECT_TRUE(n2.has_path("g/a"));
    EXPECT_EQ(n2.fetch("g/a").as_uint32(),a_val);
    EXPECT_TRUE(n2.has_path("g/b"));
    EXPECT_EQ(n2.fetch("g/b").as_uint32(),b_val);
    EXPECT_TRUE(n2.has_path("g/c"));
    EXPECT_EQ(n2.fetch("g/c").as_float64(),c_val);
    EXPECT_FALSE(n.has_path("g/d"));

    delete [] data;
}

//-----------------------------------------------------------------------------
TEST(conduit_node_paths, simple_paths)
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
    std::cout << n.schema().to_json() << std::endl; 
    const std::vector<std::string>& npaths = n.paths();
    EXPECT_EQ(npaths.size(),3);
    const std::set<std::string> npaths_set(npaths.begin(),npaths.end());
    std::set<std::string> npaths_set_expected;
    npaths_set_expected.insert("a");
    npaths_set_expected.insert("b");
    npaths_set_expected.insert("c");
    EXPECT_EQ(npaths_set,npaths_set_expected);

    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data,true);
    std::cout << n2.schema().to_json() << std::endl;
    const std::vector<std::string>& n2paths = n2.paths();
    EXPECT_EQ(n2paths.size(),1);
    EXPECT_EQ(n2paths[0],"g");

    delete [] data;
}

