// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_node_paths.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <vector>
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
    const std::vector<std::string>& npaths = n.child_names();
    EXPECT_EQ(npaths.size(),3);
    EXPECT_EQ(npaths[0],"a");
    EXPECT_EQ(npaths[1],"b");
    EXPECT_EQ(npaths[2],"c");

    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data,true);
    std::cout << n2.schema().to_json() << std::endl;
    const std::vector<std::string>& n2paths = n2.child_names();
    EXPECT_EQ(n2paths.size(),1);
    EXPECT_EQ(n2paths[0],"g");

    delete [] data;
}

//-----------------------------------------------------------------------------
TEST(conduit_node_paths, path_empty_slashes)
{ 
    Node n;
    
    n["a/b/c/d/e/f"] = 10;
    
    n.print();
    
    Node &n_sub = n["a/b/c/d/e/f"];
    EXPECT_EQ(n_sub.to_int64(),10);

    Node &n_sub_2 = n["/a/b/c/d/e/f"];
    EXPECT_EQ(n_sub_2.to_int64(),10);
    

    Node &n_sub_3 = n["/////a/b/c/d/e/f"];
    EXPECT_EQ(n_sub_3.to_int64(),10);
    

    Node &n_sub_4 = n["/////a/b/c/////d/e/f"];
    EXPECT_EQ(n_sub_4.to_int64(),10);

    n.print();
}



