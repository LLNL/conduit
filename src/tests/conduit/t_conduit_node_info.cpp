// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
    EXPECT_EQ(8,ninfo["total_strided_bytes"].to_index_t());
    EXPECT_EQ(8,ninfo["total_bytes_compact"].to_index_t());
    EXPECT_EQ(8,ninfo["total_bytes_allocated"].to_index_t());
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
    std::cout << ninfo.to_json("conduit_json") << std::endl;;
    EXPECT_EQ(72,ninfo["total_strided_bytes"].to_index_t());
    EXPECT_EQ(72,ninfo["total_bytes_compact"].to_index_t());
    EXPECT_EQ(72,ninfo["total_bytes_allocated"].to_index_t());
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
    EXPECT_EQ(4,ninfo["total_strided_bytes"].to_index_t());
    EXPECT_EQ(4,ninfo["total_bytes_allocated"].to_index_t());
    
    
    Generator g2(schema,"conduit_json",&val);
    Node n2(g2,true);
    std::cout << n2.as_uint32() << std::endl;
    EXPECT_EQ(42,val);
    
    n2.info(ninfo);
    std::cout << ninfo.to_json("conduit_json") << std::endl;;
    EXPECT_EQ(4,ninfo["total_strided_bytes"].to_index_t());
    EXPECT_EQ(0,ninfo["total_bytes_allocated"].to_index_t());
    
    
}
