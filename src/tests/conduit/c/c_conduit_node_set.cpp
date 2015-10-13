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
/// file: c_conduit_node_set.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_int_scalar)
{
    conduit_int8    i8v = -8;
    conduit_int16  i16v = -16;
    conduit_int32  i32v = -32;
    conduit_int64  i64v = -64;

    conduit_node *n = conduit_node_create();
    
    // int8
    conduit_node_set_int8(n,i8v);
    EXPECT_EQ(conduit_node_as_int8(n),i8v);
    conduit_node_print(n);
    
    // int16
    conduit_node_set_int16(n,i16v);
    EXPECT_EQ(conduit_node_as_int16(n),i16v);
    conduit_node_print(n);

    
    // int32
    conduit_node_set_int32(n,i32v);
    EXPECT_EQ(conduit_node_as_int32(n),i32v);
    conduit_node_print(n);
    
    // int64
    conduit_node_set_int64(n,i64v);
    EXPECT_EQ(conduit_node_as_int64(n),i64v);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_uint_scalar)
{
    conduit_uint8    u8v = 8;
    conduit_uint16  u16v = 16;
    conduit_uint32  u32v = 32;
    conduit_uint64  u64v = 64;

    conduit_node *n = conduit_node_create();
    
    // uint8
    conduit_node_set_uint8(n,u8v);
    EXPECT_EQ(conduit_node_as_uint8(n),u8v);
    conduit_node_print(n);
    
    // uint16
    conduit_node_set_uint16(n,u16v);
    EXPECT_EQ(conduit_node_as_uint16(n),u16v);
    conduit_node_print(n);
    
    // uint32
    conduit_node_set_uint32(n,u32v);
    EXPECT_EQ(conduit_node_as_uint32(n),u32v);
    conduit_node_print(n);
    
    // uint64
    conduit_node_set_uint64(n,u64v);
    EXPECT_EQ(conduit_node_as_uint64(n),u64v);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_float_scalar)
{
    conduit_float32 f32v =  3.1415;
    conduit_float64 f64v = -3.1415;

    conduit_node *n = conduit_node_create();
    
    // float32
    conduit_node_set_float32(n,f32v);
    EXPECT_EQ(conduit_node_as_float32(n),f32v);
    conduit_node_print(n);
    
    // float64
    conduit_node_set_float64(n,f64v);
    EXPECT_EQ(conduit_node_as_float64(n),f64v);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_int_ptr)
{
    conduit_int8    i8av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int16  i16av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int32  i32av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int64  i64av[6] = {-2,-4,-8,-16,-32,-64};
        
    conduit_node *n = conduit_node_create();
    
    // using uint8* interface
    conduit_node_set_int8_ptr(n,i8av,6);
    conduit_node_print(n);

    conduit_int8 *i8av_ptr = conduit_node_as_int8_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i8av_ptr[i],i8av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i8av_ptr[i],&i8av[i]);
    }
    EXPECT_EQ(i8av_ptr[5],-64);
    
    // using uint16* interface
    conduit_node_set_int16_ptr(n,i16av,6);
    conduit_node_print(n);
    
    conduit_int16 *i16av_ptr = conduit_node_as_int16_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i16av_ptr[i],i16av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i16av_ptr[i],&i16av[i]);
    }
    EXPECT_EQ(i16av_ptr[5],-64);
    
    // using uint32 * interface
    conduit_node_set_int32_ptr(n,i32av,6);
    conduit_node_print(n);

    conduit_int32 *i32av_ptr = conduit_node_as_int32_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i32av_ptr[i],i32av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i32av_ptr[i],&i32av[i]);
    }
    EXPECT_EQ(i32av_ptr[5],-64);
    
    // using uint64 * interface
    conduit_node_set_int64_ptr(n,i64av,6);
    conduit_node_print(n);
    
    conduit_int64 *i64av_ptr = conduit_node_as_int64_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i64av_ptr[i],i64av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i64av_ptr[i],&i64av[i]);
    }
    EXPECT_EQ(i64av_ptr[5],-64);
    
    conduit_node_destroy(n);
}



//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_uint_ptr)
{
    conduit_uint8    u8av[6] = {2,4,8,16,32,64};
    conduit_uint16  u16av[6] = {2,4,8,16,32,64};
    conduit_uint32  u32av[6] = {2,4,8,16,32,64};
    conduit_uint64  u64av[6] = {2,4,8,16,32,64};
        
    conduit_node *n = conduit_node_create();
    
    // using uint8* interface
    conduit_node_set_uint8_ptr(n,u8av,6);
    conduit_node_print(n);

    conduit_uint8 *u8av_ptr = conduit_node_as_uint8_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u8av_ptr[i],u8av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u8av_ptr[i],&u8av[i]);
    }
    EXPECT_EQ(u8av_ptr[5],64);
    
    // using uint16* interface
    conduit_node_set_uint16_ptr(n,u16av,6);
    conduit_node_print(n);
    
    conduit_uint16 *u16av_ptr = conduit_node_as_uint16_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u16av_ptr[i],u16av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u16av_ptr[i],&u16av[i]);
    }
    EXPECT_EQ(u16av_ptr[5],64);
    
    // using uint32 * interface
    conduit_node_set_uint32_ptr(n,u32av,6);
    conduit_node_print(n);

    conduit_uint32 *u32av_ptr = conduit_node_as_uint32_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u32av_ptr[i],u32av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u32av_ptr[i],&u32av[i]);
    }
    EXPECT_EQ(u32av_ptr[5],64);
    
    // using uint64 * interface
    conduit_node_set_uint64_ptr(n,u64av,6);
    conduit_node_print(n);
    
    conduit_uint64 *u64av_ptr = conduit_node_as_uint64_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u64av_ptr[i],u64av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u64av_ptr[i],&u64av[i]);
    }
    EXPECT_EQ(u64av_ptr[5],64);

    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(conduit_node_set, set_float_ptr)
{
    conduit_float32  f32av[4] = {-0.8, -1.6, -3.2, -6.4};
    conduit_float64  f64av[4] = {-0.8, -1.6, -3.2, -6.4};


        
    conduit_node *n = conduit_node_create();
    
    // float32
    conduit_node_set_float32_ptr(n,f32av,4);
    conduit_node_print(n);

    conduit_float32 *f32av_ptr = conduit_node_as_float32_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f32av_ptr[i],f32av[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&f32av_ptr[i],&f32av[i]); 
    }
    EXPECT_NEAR(f32av_ptr[3],-6.4,0.001);
    
    // float64
    conduit_node_set_float64_ptr(n,f64av,4);
    conduit_node_print(n);

    conduit_float64 *f64av_ptr = conduit_node_as_float64_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f64av_ptr[i],f64av[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);

    conduit_node_destroy(n);
}




