//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://scalability-llnl.github.io/conduit/.
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
/// file: conduit_node_update.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_update, update_simple)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    
    Node n2;
    n2["a"] = a_val + 10;
    n2["d/aa"] = a_val;
    n2["d/bb"] = b_val;
    
    n.update(n2);
    
    EXPECT_EQ(n["a"].as_uint32(),a_val+10);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);
    EXPECT_EQ(n["d/aa"].as_uint32(),a_val);
    EXPECT_EQ(n["d/bb"].as_uint32(),b_val);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_update, update_with_list)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;
    float32  d_val  = 40.0;
    
    float64 c_val_double = 60.0;

    Node n1;
    n1.append().set(a_val);
    n1.append().set(b_val);
    n1.append().set_external(&c_val);
    
    Node n2;
    n2.append().set(a_val*2);
    n2.append().set(b_val*2);
    n2.append().set(c_val*2);
    n2.append().set(d_val);

    
    n1.update(n2);
    EXPECT_EQ(n1[0].as_uint32(),a_val*2);
    EXPECT_EQ(n1[1].as_uint32(),b_val*2);
    EXPECT_EQ(n1[2].as_float64(),c_val_double);
    EXPECT_EQ(n1[3].as_float32(),d_val);

    // we did something tricky with set external for c_val, see if it worked.
    EXPECT_NEAR(c_val,c_val_double,0.001);
}


//-----------------------------------------------------------------------------
TEST(conduit_node_update, update_realloc_like)
{
    std::vector<uint32> vals;
    for(index_t i=0;i<10;i++)
    {
        vals.push_back(i);
    }

    Node n;
    n["a"].set(vals);
    
    Node n2;
    n2["a"].set(DataType::uint32(15));
    // zero out the buffer just to be safe for this unit test
    memset(n2["a"].data_ptr(),0,sizeof(uint32)*15);
    
    n2.update(n);

    uint32 *n_v_ptr  = n["a"].as_uint32_ptr();    
    uint32 *n2_v_ptr = n2["a"].as_uint32_ptr();

    for(index_t i=0;i<10;i++)
    {
        EXPECT_EQ(n_v_ptr[i],n2_v_ptr[i]);
    }    
    
    for(index_t i=10;i<15;i++)
    {
        EXPECT_EQ(n2_v_ptr[i],0); // assumes zeroed-alloc
    }
}


