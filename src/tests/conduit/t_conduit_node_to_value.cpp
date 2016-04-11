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
/// file: conduit_node_to_value.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, string_to_scalar)
{
    Node n;
    n.set("127");
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}


//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, uint8_to_scalar)
{
    Node n;
    n.set_uint8(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, uint16_to_scalar)
{
    Node n;
    n.set_uint16(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, uint32_to_scalar)
{
    Node n;
    n.set_uint32(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, uint64_to_scalar)
{
    Node n;
    n.set_uint64(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, int8_to_scalar)
{
    Node n;
    n.set_int8(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, int16_to_scalar)
{
    Node n;
    n.set_int16(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, int32_to_scalar)
{
    Node n;
    n.set_int32(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, int64_to_scalar)
{
    Node n;
    n.set_int64(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}


//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, float32_to_scalar)
{
    Node n;
    n.set_float32(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_value, float64_to_scalar)
{
    Node n;
    n.set_float64(127);
    
    EXPECT_EQ( n.to_uint8(),127);
    EXPECT_EQ( n.to_uint16(),127);
    EXPECT_EQ( n.to_uint32(),127);
    EXPECT_EQ( n.to_uint64(),127);

    EXPECT_EQ( n.to_int8(),127);
    EXPECT_EQ( n.to_int16(),127);
    EXPECT_EQ( n.to_int32(),127);
    EXPECT_EQ( n.to_int64(),127);

    EXPECT_NEAR( n.to_float32(),127,0.0001);
    EXPECT_NEAR( n.to_float64(),127,0.0001);

    EXPECT_EQ( n.to_unsigned_char(),127);
    EXPECT_EQ( n.to_unsigned_short(),127);
    EXPECT_EQ( n.to_unsigned_int(),127);
    EXPECT_EQ( n.to_unsigned_long(),127);

    EXPECT_EQ( n.to_char(),127);
    EXPECT_EQ( n.to_short(),127);
    EXPECT_EQ( n.to_int(),127);
    EXPECT_EQ( n.to_long(),127);
    
    EXPECT_NEAR( n.to_float(),127,0.0001);
    EXPECT_NEAR( n.to_double(),127,0.0001);
}



