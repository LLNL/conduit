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
/// file: conduit_node_compact.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

TEST(compact_1, conduit_node_compact)
{

    uint32   vals[] = {10,20,30,40,50,60,70,80,90,100};

    
    Generator g("{vals: {dtype:uint32, length:5, stride:8}}",vals);
    Node n(g);

    EXPECT_EQ(40,n.total_bytes());
    EXPECT_EQ(20,n.total_bytes_compact());
    n.print();
    Node nc;
    n.compact_to(nc);
    nc.schema().print();
    nc.print_detailed();
    EXPECT_EQ(20,nc.total_bytes());
    EXPECT_EQ(20,nc.total_bytes_compact());

    uint32_array n_arr  = n["vals"].as_uint32_array();
    uint32_array nc_arr = nc["vals"].as_uint32_array();
    EXPECT_EQ(n_arr[2],nc_arr[2]);
}

TEST(compact_2, conduit_node_compact)
{

    float64 vals[] = { 100.0,-100.0,200.0,-200.0,300.0,-300.0,400.0,-400.0,500.0,-500.0};
    Generator g1("{dtype: float64, length: 5, stride: 16}",vals);
    Generator g2("{dtype: float64, length: 5, stride: 16, offset:8}",vals);

    Node n1(g1);
    n1.print();

    Node n2(g2);
    n2.print();
    
    Node ninfo;
    n1.info(ninfo);
    ninfo.print();

    Node n1c;
    n1.compact_to(n1c);

    n1c.schema().print();
    n1c.print_detailed();
    n1c.info(ninfo);
    ninfo.print();

    float64_array n1_arr  = n1.as_float64_array();
    float64_array n1c_arr = n1c.as_float64_array();
    for(index_t i=0;i<5;i++)
    {
        EXPECT_EQ(n1_arr[i],n1c_arr[i]);
    }    
}