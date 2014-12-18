//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// file: conduit_node_info.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;


TEST(info_simple_1, conduit_node_info)
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
    std::cout << ninfo.to_json(true) << std::endl;;
    EXPECT_EQ(8,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(8,ninfo["total_bytes_compact"].to_index_t());
    EXPECT_EQ(8,ninfo["total_bytes_alloced"].to_index_t());
    EXPECT_EQ(0,ninfo["total_bytes_mmaped"].to_index_t());
}


TEST(info_simple_2, conduit_node_info)
{
    std::string pure_json ="{a:[0,1,2,3,4],b:[0.0,1.1,2.2,3.3]}";
    Generator g(pure_json,"json");
    Node n(g);
    Node ninfo;
    n.info(ninfo);
    std::cout << ninfo.to_json(true,2) << std::endl;;
    EXPECT_EQ(72,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(72,ninfo["total_bytes_compact"].to_index_t());
    EXPECT_EQ(72,ninfo["total_bytes_alloced"].to_index_t());
    EXPECT_EQ(0,ninfo["total_bytes_mmaped"].to_index_t());
}


TEST(info_simple_3, conduit_node_info)
{
    uint32   val=0;

    std::string schema ="{dtype: uint32, value:42}";
    // TODO: check for "unit32" , bad spelling!
    Generator g(schema);
    Node n(g);
    std::cout << n.as_uint32() << std::endl;
    Node ninfo;
    n.info(ninfo);
    EXPECT_EQ(42,n.as_uint32());
    EXPECT_EQ(4,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(4,ninfo["total_bytes_alloced"].to_index_t());
    
    
    Generator g2(schema,&val);
    Node n2(g2);
    std::cout << n2.as_uint32() << std::endl;
    EXPECT_EQ(42,val);
    
    n2.info(ninfo);
    std::cout << ninfo.to_json(true,2) << std::endl;;
    EXPECT_EQ(4,ninfo["total_bytes"].to_index_t());
    EXPECT_EQ(0,ninfo["total_bytes_alloced"].to_index_t());
    
    
}
