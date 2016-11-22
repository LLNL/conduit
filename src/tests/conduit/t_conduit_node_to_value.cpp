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
// For details, see: http://llnl.github.io/conduit/.
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

//-----------------------------------------------------------------------------
// check default values from as_zzz, by avoiding exceptions on warn
//----------------------------------------------------------------------------- 

//-----------------------------------------------------------------------------
void 
print_msg(const std::string &msg,
          const std::string &file,
          int line)
{
    std::cout << "File:"    << file << std::endl;
    std::cout << "Line:"    << line << std::endl;
    std::cout << "Message:" << msg  << std::endl;
}

//-----------------------------------------------------------------------------
void 
my_warning_handler(const std::string &msg,
                   const std::string &file,
                   int line)
{
    print_msg(msg,file,line);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_set, get_from_empty)
{
    // override warning handler so we can check the default
    // return values on dtype mismatch
    conduit::utils::set_warning_handler(my_warning_handler);
    
    Node n;// empty

    // bw-style types (pointers)
    int8     *int8_ptr    = n.value();
    int16    *int16_ptr   = n.value();
    int32    *int32_ptr   = n.value();
    int64    *int64_ptr   = n.value();
    
    uint8     *uint8_ptr    = n.value();
    uint16    *uint16_ptr   = n.value();
    uint32    *uint32_ptr   = n.value();
    uint64    *uint64_ptr   = n.value();
    
    float32  *float32_ptr = n.value();
    float64  *float64_ptr = n.value();

    EXPECT_TRUE(int8_ptr == NULL);
    EXPECT_TRUE(int16_ptr == NULL);
    EXPECT_TRUE(int32_ptr == NULL);
    EXPECT_TRUE(int64_ptr == NULL);

    EXPECT_TRUE(uint8_ptr == NULL);
    EXPECT_TRUE(uint16_ptr == NULL);
    EXPECT_TRUE(uint32_ptr == NULL);
    EXPECT_TRUE(uint64_ptr == NULL);
    
    EXPECT_TRUE(float32_ptr == NULL);
    EXPECT_TRUE(float64_ptr == NULL);

    // bw-style types (scalars)

    int8  int8_val  = n.value();
    int16 int16_val = n.value();
    int32 int32_val = n.value();
    int64 int64_val = n.value();

    uint8  uint8_val  = n.value();
    uint16 uint16_val = n.value();
    uint32 uint32_val = n.value();
    uint64 uint64_val = n.value();
    
    float32 float32_val = n.value();
    float64 float64_val = n.value();
    
    EXPECT_EQ(int8_val,0);
    EXPECT_EQ(int16_val,0);
    EXPECT_EQ(int32_val,0);
    EXPECT_EQ(int64_val,0);

    EXPECT_EQ(uint8_val,0);
    EXPECT_EQ(uint16_val,0);
    EXPECT_EQ(uint32_val,0);
    EXPECT_EQ(uint64_val,0);

    EXPECT_EQ(float32_val,0);
    EXPECT_EQ(float64_val,0);

    // c native types (pointers)
    char   *char_ptr   = n.value();
    short  *short_ptr  = n.value();
    int    *int_ptr    = n.value();
    long   *long_ptr   = n.value();

    unsigned char   *uchar_ptr  = n.value();
    unsigned short  *ushort_ptr = n.value();
    unsigned int    *uint_ptr   = n.value();
    unsigned long   *ulong_ptr  = n.value();

    float  *float_ptr  = n.value();
    double *double_ptr = n.value();

    EXPECT_TRUE(char_ptr == NULL);
    EXPECT_TRUE(short_ptr == NULL);
    EXPECT_TRUE(int_ptr == NULL);
    EXPECT_TRUE(long_ptr == NULL);

    EXPECT_TRUE(uchar_ptr == NULL);
    EXPECT_TRUE(ushort_ptr == NULL);
    EXPECT_TRUE(uint_ptr == NULL);
    EXPECT_TRUE(ulong_ptr == NULL);
    
    EXPECT_TRUE(float_ptr == NULL);
    EXPECT_TRUE(double_ptr == NULL);
    
    // c native types (scalars)
    signed char  char_val  = n.value();
    
    short  short_val = n.value();
    int    int_val   = n.value();
    long   long_val  = n.value();

    unsigned char   uchar_val  = n.value();
    unsigned short  ushort_val = n.value();
    unsigned int    uint_val   = n.value();
    unsigned long   ulong_val  = n.value();
    
    float  float_val  = n.value();
    double double_val = n.value();

    EXPECT_EQ(char_val,0);
    EXPECT_EQ(short_val,0);
    EXPECT_EQ(int_val,0);
    EXPECT_EQ(long_val,0);


    EXPECT_EQ(uchar_val,0);
    EXPECT_EQ(ushort_val,0);
    EXPECT_EQ(uint_val,0);
    EXPECT_EQ(ulong_val,0);
    
    EXPECT_EQ(float_val,0);
    EXPECT_EQ(double_val,0);

    // reset warning handler to default
    conduit::utils::set_warning_handler(conduit::utils::default_warning_handler);
}



