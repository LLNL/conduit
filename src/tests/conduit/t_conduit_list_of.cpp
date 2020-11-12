// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_list_of.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_list_of, simple )
{
    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[16];
    memcpy(&data[0],&a1_val,4);
    memcpy(&data[4],&b1_val,4);
    memcpy(&data[8],&a2_val,4);
    memcpy(&data[12],&b2_val,4);
    
    std::string jschema = "{\"dtype\":{\"a\":\"int32\",\"b\":\"int32\"},\"length\":2}";
	Schema sch(jschema);
    Node n(sch,data,true);
    std::cout << n.schema().to_json() << std::endl;
    std::cout << n.to_json() << std::endl;
    
    std::cout <<  n[0]["a"].as_int32() << std::endl;
    std::cout <<  n[1]["a"].as_int32() << std::endl;

    std::cout <<  n[0]["b"].as_int32() << std::endl;
    std::cout <<  n[1]["b"].as_int32() << std::endl;

    EXPECT_EQ(n[0]["a"].as_int32(), a1_val);
    EXPECT_EQ(n[1]["a"].as_int32(), a2_val);

    EXPECT_EQ(n[0]["b"].as_int32(), b1_val);
    EXPECT_EQ(n[1]["b"].as_int32(), b2_val);

    delete [] data;
}

//-----------------------------------------------------------------------------
TEST(conduit_list_of, path_ref)
{
    uint32  len_val = 2;
    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[20];
    memcpy(&data[0],&len_val,4);    
    memcpy(&data[4],&a1_val,4);
    memcpy(&data[8],&b1_val,4);
    memcpy(&data[12],&a2_val,4);
    memcpy(&data[16],&b2_val,4);
    
    std::string jschema = "{ \"list_length\": \"uint32\", \"values\":{\"dtype\":{\"a\":\"int32\",\"b\":\"int32\"},\"length\":{\"reference\":\"../list_length\"}}}";
	
    //For ref case, we will have to pass the json directly
    //
    Node n(jschema,data,true);
    std::cout << n.schema().to_json() << std::endl;
    std::cout << n.to_json() << std::endl;

    std::cout <<  n["list_length"].as_uint32() << std::endl;

    std::cout <<  n["values"][0]["a"].as_int32() << std::endl;
    std::cout <<  n["values"][1]["a"].as_int32() << std::endl;

    std::cout <<  n["values"][0]["b"].as_int32() << std::endl;
    std::cout <<  n["values"][1]["b"].as_int32() << std::endl;

    EXPECT_EQ(n["list_length"].as_uint32(), len_val);
    EXPECT_EQ(n["values"][0]["a"].as_int32(), a1_val);
    EXPECT_EQ(n["values"][1]["a"].as_int32(), a2_val);

    EXPECT_EQ(n["values"][0]["b"].as_int32(), b1_val);
    EXPECT_EQ(n["values"][1]["b"].as_int32(), b2_val);

    delete [] data;
}
    
    

//-----------------------------------------------------------------------------
TEST(conduit_list_of, without_json)
{
    uint32  len_val = 2;
    int32   a1_val  = 10;
    int32   b1_val  = 20;
    
    int32   a2_val  = -10;
    int32   b2_val  = -20;
    
    char *data = new char[16];
    memcpy(&data[0],&a1_val,4);    
    memcpy(&data[4],&b1_val,4);
    memcpy(&data[8],&a2_val,4);
    memcpy(&data[12],&b2_val,4);
    
    Node n;
    
    Schema entry_s;
    entry_s["a"].set(DataType::int32());
    entry_s["b"].set(DataType::int32());
    
    n["list_length"] = len_val;
    n["values"].list_of_external(data,entry_s,2);
    
    std::cout << n.schema().to_json() << std::endl;
    std::cout << n.to_json() << std::endl;

    std::cout <<  n["list_length"].as_uint32() << std::endl;

    std::cout <<  n["values"][0]["a"].as_int32() << std::endl;
    std::cout <<  n["values"][1]["a"].as_int32() << std::endl;

    std::cout <<  n["values"][0]["b"].as_int32() << std::endl;
    std::cout <<  n["values"][1]["b"].as_int32() << std::endl;

    EXPECT_EQ(n["list_length"].as_uint32(), len_val);
    EXPECT_EQ(n["values"][0]["a"].as_int32(), a1_val);
    EXPECT_EQ(n["values"][1]["a"].as_int32(), a2_val);

    EXPECT_EQ(n["values"][0]["b"].as_int32(), b1_val);
    EXPECT_EQ(n["values"][1]["b"].as_int32(), b2_val);


    // test the non-external case:
    Node n2;
    n2.list_of(entry_s,2);
    n2.update(n["values"]);
    n2.print();
    
    // we should only have one allocation 
    Node ninfo;
    n2.info(ninfo);
    ninfo.print();
    

    // test if n2 was actually created in a contiguous way
    Node n3;
    n3.set_external((int32*)n2.data_ptr(),4);
    n3.print();
    
    int32 *n3_ptr = n3.value();
    
    EXPECT_EQ(n3_ptr[0], a1_val);
    EXPECT_EQ(n3_ptr[1], b1_val);
    EXPECT_EQ(n3_ptr[2], a2_val);
    EXPECT_EQ(n3_ptr[3], b2_val);

    delete [] data;
}
    
    

//-----------------------------------------------------------------------------
TEST(conduit_list_of, dtype)
{
    Node n;
    n.list_of(DataType::empty(),5);
    n.info().print();
    n.print();
    EXPECT_TRUE(n.dtype().is_list());
    EXPECT_EQ(n.number_of_children(),5);
    
    
    n.list_of(DataType::float64(2),5);
    n.print();
    n.info().print();
    EXPECT_EQ(n.allocated_bytes(), 5 * 2 * 8);
    

    NodeIterator itr = n.children();
    while(itr.has_next())
    {
        float64_array vals = itr.next().value();
        vals[0] = 10;
        vals[1] = 20;
    }    
    
    n.print();
    
    
}
