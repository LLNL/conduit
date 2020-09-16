// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_to_string.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_to_string, simple_1)
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
    n.schema().print();
    n.print_detailed();
    EXPECT_EQ(std::string("{\"a\": 10,\"b\": 20,\"c\": 30.0}"),n.to_json("json",0,0,"",""));
    


    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data,true);
    n2.schema().print();
    n.print_detailed();
    EXPECT_EQ(std::string("{\"g\": {\"a\": 10,\"b\": 20,\"c\": 30.0}}"),n2.to_json("json",0,0,"",""));

    delete [] data;
}

