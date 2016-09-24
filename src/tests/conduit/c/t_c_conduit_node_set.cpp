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
    conduit_float32 f32v =  3.1415f;
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
TEST(conduit_node_set, set_bitwidth_float_ptr)
{
    conduit_float32  f32av[4] = {-0.8f, -1.6f, -3.2f, -6.4f};
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
    
    
    // float32 detailed
    conduit_node_set_float32_ptr_detailed(n,
                                          f32av,
                                          4,
                                          0,
                                          sizeof(conduit_float32),
                                          sizeof(conduit_float32),
                                          CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    f32av_ptr = conduit_node_as_float32_ptr(n);
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

    // float64 detailed
    conduit_node_set_float64_ptr_detailed(n,
                                          f64av,
                                          4,
                                          0,
                                          sizeof(conduit_float64),
                                          sizeof(conduit_float64),
                                          CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    f64av_ptr = conduit_node_as_float64_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f64av_ptr[i],f64av[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);


    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
// c style tests
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_int_scalar)
{
    char   icv = -8;
    short  isv = -16;
    int    iiv = -32;
    long   ilv = -64;

    conduit_node *n = conduit_node_create();
    
    // char
    conduit_node_set_char(n,icv);
    EXPECT_EQ(conduit_node_as_char(n),icv);
    conduit_node_print(n);
    
    // short
    conduit_node_set_short(n,isv);
    EXPECT_EQ(conduit_node_as_short(n),isv);
    conduit_node_print(n);

    
    // int
    conduit_node_set_int(n,iiv);
    EXPECT_EQ(conduit_node_as_int(n),iiv);
    conduit_node_print(n);
    
    // long
    conduit_node_set_long(n,ilv);
    EXPECT_EQ(conduit_node_as_long(n),ilv);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_uint_scalar)
{
    unsigned char   ucv = -8;
    unsigned short  usv = -16;
    unsigned int    uiv = -32;
    unsigned long   ulv = -64;

    conduit_node *n = conduit_node_create();
    
    // char
    conduit_node_set_unsigned_char(n,ucv);
    EXPECT_EQ(conduit_node_as_unsigned_char(n),ucv);
    conduit_node_print(n);
    
    // short
    conduit_node_set_unsigned_short(n,usv);
    EXPECT_EQ(conduit_node_as_unsigned_short(n),usv);
    conduit_node_print(n);

    
    // int
    conduit_node_set_unsigned_int(n,uiv);
    EXPECT_EQ(conduit_node_as_unsigned_int(n),uiv);
    conduit_node_print(n);
    
    // long
    conduit_node_set_unsigned_long(n,ulv);
    EXPECT_EQ(conduit_node_as_unsigned_long(n),ulv);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_float_scalar)
{
    float  fv =  3.1415f;
    double dv = -3.1415;

    conduit_node *n = conduit_node_create();
    
    // float32
    conduit_node_set_float(n,fv);
    EXPECT_EQ(conduit_node_as_float(n),fv);
    conduit_node_print(n);
    
    // float64
    conduit_node_set_double(n,dv);
    EXPECT_EQ(conduit_node_as_double(n),dv);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_int_ptr)
{
    char  icav[6] = {-2,-4,-8,-16,-32,-64};
    short isav[6] = {-2,-4,-8,-16,-32,-64};
    int   iiav[6] = {-2,-4,-8,-16,-32,-64};
    long  ilav[6] = {-2,-4,-8,-16,-32,-64};
        
    conduit_node *n = conduit_node_create();
    
    // using char* interface
    conduit_node_set_char_ptr(n,icav,6);
    conduit_node_print(n);

    char *icav_ptr = conduit_node_as_char_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(icav_ptr[i],icav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&icav_ptr[i],&icav[i]);
    }
    EXPECT_EQ(icav_ptr[5],-64);
    
    // using short* interface
    conduit_node_set_short_ptr(n,isav,6);
    conduit_node_print(n);
    
    short *isav_ptr = conduit_node_as_short_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(isav_ptr[i],isav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&isav_ptr[i],&isav[i]);
    }
    EXPECT_EQ(isav_ptr[5],-64);
    
    // using int* interface
    conduit_node_set_int_ptr(n,iiav,6);
    conduit_node_print(n);

    int *iiav_ptr = conduit_node_as_int_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(iiav_ptr[i],iiav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&iiav_ptr[i],&iiav[i]);
    }
    EXPECT_EQ(iiav_ptr[5],-64);
    
    // using long * interface
    conduit_node_set_long_ptr(n,ilav,6);
    conduit_node_print(n);
    
    long *ilav_ptr = conduit_node_as_long_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ilav_ptr[i],ilav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&ilav_ptr[i],&ilav[i]);
    }
    EXPECT_EQ(ilav_ptr[5],-64);
    
    conduit_node_destroy(n);
}



//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_uint_ptr)
{
    unsigned char   ucav[6] = {2,4,8,16,32,64};
    unsigned short  usav[6] = {2,4,8,16,32,64};
    unsigned int    uiav[6] = {2,4,8,16,32,64};
    unsigned long   ulav[6] = {2,4,8,16,32,64};
        
    conduit_node *n = conduit_node_create();
    
    // using unsigned char* interface
    conduit_node_set_unsigned_char_ptr(n,ucav,6);
    conduit_node_print(n);

    unsigned char *ucav_ptr = conduit_node_as_unsigned_char_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ucav_ptr[i],ucav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&ucav_ptr[i],&ucav[i]);
    }
    EXPECT_EQ(ucav_ptr[5],64);
    
    // using unsigned short* interface
    conduit_node_set_unsigned_short_ptr(n,usav,6);
    conduit_node_print(n);
    
    unsigned short *usav_ptr = conduit_node_as_unsigned_short_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(usav_ptr[i],usav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&usav_ptr[i],&usav[i]);
    }
    EXPECT_EQ(usav_ptr[5],64);
    
    // using unsigned int * interface
    conduit_node_set_unsigned_int_ptr(n,uiav,6);
    conduit_node_print(n);

    unsigned int *uiav_ptr = conduit_node_as_unsigned_int_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(uiav_ptr[i],uiav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&uiav_ptr[i],&uiav[i]);
    }
    EXPECT_EQ(uiav_ptr[5],64);
    
    // using unsigned long * interface
    conduit_node_set_unsigned_long_ptr(n,ulav,6);
    conduit_node_print(n);
    
    unsigned long *ulav_ptr = conduit_node_as_unsigned_long_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ulav_ptr[i],ulav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&ulav_ptr[i],&ulav[i]);
    }
    EXPECT_EQ(ulav_ptr[5],64);

    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
TEST(conduit_node_set, set_native_float_ptr)
{
    float   fav[4] = {-0.8f, -1.6f, -3.2f, -6.4f};
    double  dav[4] = {-0.8, -1.6, -3.2, -6.4};

    conduit_node *n = conduit_node_create();
    
    // float*
    conduit_node_set_float_ptr(n,fav,4);
    conduit_node_print(n);

    float *fav_ptr = conduit_node_as_float_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(fav_ptr[i],fav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // float32 detailed
    conduit_node_set_float_ptr_detailed(n,
                                        fav,
                                        4,
                                        0,
                                        sizeof(float),
                                        sizeof(float),
                                        CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    fav_ptr = conduit_node_as_float_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(fav_ptr[i],fav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // double
    conduit_node_set_double_ptr(n,dav,4);
    conduit_node_print(n);

    double *dav_ptr = conduit_node_as_double_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(dav_ptr[i],dav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);

    // float64 detailed
    conduit_node_set_double_ptr_detailed(n,
                                         dav,
                                         4,
                                         0,
                                         sizeof(double),
                                         sizeof(double),
                                         CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    dav_ptr = conduit_node_as_double_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(dav_ptr[i],dav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);


    conduit_node_destroy(n);
}


