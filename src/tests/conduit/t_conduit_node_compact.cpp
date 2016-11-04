//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
/// file: conduit_node_compact.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_compact, compact_1)
{

    uint32   vals[] = {10,20,30,40,50,60,70,80,90,100};

    
    Generator g("{vals: {dtype:uint32, length:5, stride:8}}",
                "conduit_json",
                vals);
    Node n(g,true);

    EXPECT_EQ(36,n.total_strided_bytes());
    EXPECT_EQ(20,n.total_bytes_compact());
    n.print();
    Node nc;
    n.compact_to(nc);
    nc.schema().print();
    nc.print_detailed();
    EXPECT_EQ(20,nc.total_strided_bytes());
    EXPECT_EQ(20,nc.total_bytes_compact());

    uint32_array n_arr  = n["vals"].as_uint32_array();
    uint32_array nc_arr = nc["vals"].as_uint32_array();
    EXPECT_EQ(n_arr[2],nc_arr[2]);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compact, compact_2)
{

    float64 vals[] = { 100.0,-100.0,200.0,-200.0,300.0,-300.0,400.0,-400.0,500.0,-500.0};
    Generator g1("{dtype: float64, length: 5, stride: 16}",
                 "conduit_json",
                 vals);
    
    Generator g2("{dtype: float64, length: 5, stride: 16, offset:8}",
                 "conduit_json",
                  vals);

    Node n1(g1,true);
    n1.print();

    Node n2(g2,true);
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

//-----------------------------------------------------------------------------
TEST(conduit_node_compact, compact_3)
{

    float64 vals[] = { 100.0,-100.0,200.0,-200.0,300.0,-300.0,400.0,-400.0,500.0,-500.0};

    Node n;
    n["a"].set_external(vals,10);
    n.print();

    Node nc;
    n.compact_to(nc);
    nc.schema().print();
    nc.print_detailed();
    nc.info().print();

    float64_array n_arr  = n["a"].as_float64_array();
    float64_array nc_arr = nc["a"].as_float64_array();
    for(index_t i=0;i<10;i++)
    {
        EXPECT_EQ(n_arr[i],nc_arr[i]);
    }
}
