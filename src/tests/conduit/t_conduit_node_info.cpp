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
/// file: conduit_node_info.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_info, simple_1)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    std::cout << n.to_json();
    
    Node ninfo;
    n.info(ninfo);
    std::cout << ninfo.to_json() << std::endl;;
    EXPECT_EQ(8,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(8,ninfo["total_bytes_compact"].to_index_t());
    EXPECT_EQ(8,ninfo["total_bytes_alloced"].to_index_t());
    EXPECT_EQ(0,ninfo["total_bytes_mmaped"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_node_info, simple_2)
{
    std::string pure_json ="{a:[0,1,2,3,4],b:[0.0,1.1,2.2,3.3]}";
    Generator g(pure_json,"json");
    Node n(g,true);
    Node ninfo;
    n.info(ninfo);
    std::cout << ninfo.to_json("conduit") << std::endl;;
    EXPECT_EQ(72,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(72,ninfo["total_bytes_compact"].to_index_t());
    EXPECT_EQ(72,ninfo["total_bytes_alloced"].to_index_t());
    EXPECT_EQ(0,ninfo["total_bytes_mmaped"].to_index_t());
}

//-----------------------------------------------------------------------------
TEST(conduit_node_info, simple_3)
{
    uint32   val=0;

    std::string schema ="{dtype: uint32, value:42}";
    // TODO: check for "unit32" , bad spelling!
    Generator g(schema);
    Node n(g,true);
    std::cout << n.as_uint32() << std::endl;
    Node ninfo;
    n.info(ninfo);
    EXPECT_EQ(42,n.as_uint32());
    EXPECT_EQ(4,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(4,ninfo["total_bytes_alloced"].to_index_t());
    
    
    Generator g2(schema,&val);
    Node n2(g2,true);
    std::cout << n2.as_uint32() << std::endl;
    EXPECT_EQ(42,val);
    
    n2.info(ninfo);
    std::cout << ninfo.to_json("conduit") << std::endl;;
    EXPECT_EQ(4,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(0,ninfo["total_bytes_alloced"].to_index_t());
    
    
}
