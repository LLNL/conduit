// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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

    Generator g("{dtype: float64, length: 5, stride: 16, offset:8}",
                "conduit_json",
                vals);

    Node n(g,true);


    EXPECT_EQ(n.info()["total_strided_bytes"].to_uint64(),72);
    
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
    
    EXPECT_EQ(nc.info()["total_strided_bytes"].to_uint64(),40);
    
}


//-----------------------------------------------------------------------------
// Test for https://github.com/LLNL/conduit/issues/226
//
// Adapted from: test from @tomvierjahn's fork (linked above)
//-----------------------------------------------------------------------------

TEST(conduit_serialize, seralize_multiple)
{
    Node node;
    node["a/b/c"].set((uint8)1);
    node["a/b/d"].set((uint8)2);
    node["a/e"].set((uint8)3);
    
    std::string schema;
    std::vector<uint8> bytes;
    
    Schema s_compact;
    // serialize
    node.schema().compact_to(s_compact);
    schema = s_compact.to_json();
    node.serialize(bytes);
    
    const std::vector<uint8> bytes_serialization1(bytes);
    const std::string schema_serialization1(schema);

    
    Node second_node;
    second_node.set_data_using_schema(Schema(schema), bytes.data());
    EXPECT_EQ(node["a/b/c"].as_uint8(), second_node["a/b/c"].as_uint8());
    EXPECT_EQ(node["a/b/d"].as_uint8(), second_node["a/b/d"].as_uint8());
    EXPECT_EQ(node["a/e"].as_uint8(), second_node["a/e"].as_uint8());
    
    // serialize again 
    second_node.schema().compact_to(s_compact);
    schema = s_compact.to_json();
    second_node.serialize(bytes);
    
    const std::vector<uint8> bytes_serialization2(bytes);
    const std::string schema_serialization2(schema);
    
    EXPECT_EQ(bytes_serialization1, bytes_serialization2);
    EXPECT_EQ(schema_serialization1, schema_serialization2);
    
    Node third_node;
    third_node.set_data_using_schema(Schema(schema), bytes.data());
    
    node.schema().print();
    node.print();
    
    second_node.schema().print();
    second_node.print();
    
    
    third_node.schema().print();
    third_node.print();


    EXPECT_EQ(node["a/b/c"].as_uint8(), third_node["a/b/c"].as_uint8());
    EXPECT_EQ(node["a/b/d"].as_uint8(), third_node["a/b/d"].as_uint8());
    EXPECT_EQ(node["a/e"].as_uint8(), third_node["a/e"].as_uint8());
}


