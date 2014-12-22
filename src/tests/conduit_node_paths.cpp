//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_node_paths.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

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
    Node n(schema,data);
    std::cout << n.schema().to_json() <<std::endl; 
    
    EXPECT_TRUE(n.has_path("a"));
    EXPECT_EQ(n.fetch("a").as_uint32(),a_val);
    
    EXPECT_TRUE(n.has_path("b"));
    EXPECT_EQ(n.fetch("b").as_uint32(),b_val);
    
    EXPECT_TRUE(n.has_path("c"));
    EXPECT_EQ(n.fetch("c").as_float64(),c_val);

    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data);
    std::cout << n2.schema().to_json() <<std::endl; 
    EXPECT_TRUE(n2.has_path("g/a"));
    EXPECT_EQ(n2.fetch("g/a").as_uint32(),a_val);
    EXPECT_TRUE(n2.has_path("g/b"));
    EXPECT_EQ(n2.fetch("g/b").as_uint32(),b_val);
    EXPECT_TRUE(n2.has_path("g/c"));
    EXPECT_EQ(n2.fetch("g/c").as_float64(),c_val);

    EXPECT_FALSE(n.has_path("g/d"));
}

