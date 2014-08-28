/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: conduit_to_string.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

TEST(to_string_simple_1, conduit_to_string)
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
    n.schema().print();
    n.print_detailed();
    EXPECT_EQ(std::string("{\"a\": 10,\"b\": 20,\"c\": 30}"),n.to_json(false,0,0,"",""));
    


    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data);
    n2.schema().print();
    n.print_detailed();
    EXPECT_EQ(std::string("{\"g\": {\"a\": 10,\"b\": 20,\"c\": 30}}"),n2.to_json(false,0,0,"",""));
}

