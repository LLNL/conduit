// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: c_conduit_node_set.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// bitwidth style tests
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// set and set_path tests
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_int_scalar)
{
    conduit_int8    i8v = -8;
    conduit_int16  i16v = -16;
    conduit_int32  i32v = -32;
    conduit_int64  i64v = -64;

    conduit_node *n = conduit_node_create();
    
    //------------
    // set 
    //------------
    
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
    
    //------------
    // set_path
    //------------
    
    // int8
    conduit_node_set_path_int8(n,"i8",i8v);
    EXPECT_EQ(conduit_node_fetch_path_as_int8(n,"i8"),i8v);
    conduit_node_print(n);
    
    // int16
    conduit_node_set_path_int16(n,"i16",i16v);
    EXPECT_EQ(conduit_node_fetch_path_as_int16(n,"i16"),i16v);
    conduit_node_print(n);

    
    // int32
    conduit_node_set_path_int32(n,"i32",i32v);
    EXPECT_EQ(conduit_node_fetch_path_as_int32(n,"i32"),i32v);
    conduit_node_print(n);
    
    // int64
    conduit_node_set_path_int64(n,"i64",i64v);
    EXPECT_EQ(conduit_node_fetch_path_as_int64(n,"i64"),i64v);
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
    
    //------------
    // set 
    //------------
    
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
    
    //------------
    // set_path
    //------------
    
    // uint8
    conduit_node_set_path_uint8(n,"u8",u8v);
    EXPECT_EQ(conduit_node_fetch_path_as_uint8(n,"u8"),u8v);
    conduit_node_print(n);
    
    // uint16
    conduit_node_set_path_uint16(n,"u16",u16v);
    EXPECT_EQ(conduit_node_fetch_path_as_uint16(n,"u16"),u16v);
    conduit_node_print(n);
    
    // uint32
    conduit_node_set_path_uint32(n,"u32",u32v);
    EXPECT_EQ(conduit_node_fetch_path_as_uint32(n,"u32"),u32v);
    conduit_node_print(n);
    
    // uint64
    conduit_node_set_path_uint64(n,"u64",u64v);
    EXPECT_EQ(conduit_node_fetch_path_as_uint64(n,"u64"),u64v);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_float_scalar)
{
    conduit_float32 f32v =  3.1415f;
    conduit_float64 f64v = -3.1415;

    conduit_node *n = conduit_node_create();
    
    //------------
    // set 
    //------------
    
    // float32
    conduit_node_set_float32(n,f32v);
    EXPECT_EQ(conduit_node_as_float32(n),f32v);
    conduit_node_print(n);
    
    // float64
    conduit_node_set_float64(n,f64v);
    EXPECT_EQ(conduit_node_as_float64(n),f64v);
    conduit_node_print(n);
    
    //------------
    // set 
    //------------
    
    // float32
    conduit_node_set_path_float32(n,"f32",f32v);
    EXPECT_EQ(conduit_node_fetch_path_as_float32(n,"f32"),f32v);
    conduit_node_print(n);
    
    // float64
    conduit_node_set_path_float64(n,"f64",f64v);
    EXPECT_EQ(conduit_node_fetch_path_as_float64(n,"f64"),f64v);
    conduit_node_print(n);
    
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
// set and set_external tests
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_bitwidth_int_ptr)
{
    conduit_int8    i8av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int16  i16av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int32  i32av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int64  i64av[6] = {-2,-4,-8,-16,-32,-64};
        
    conduit_node *n = conduit_node_create();
    
    //--------------
    // set
    //--------------
    
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
    
    //--------------
    // set_external
    //--------------
    
    // using uint8* interface
    conduit_node_set_external_int8_ptr(n,i8av,6);
    conduit_node_print(n);

    i8av_ptr = conduit_node_as_int8_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i8av_ptr[i],i8av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i8av_ptr[i],&i8av[i]);
    }
    EXPECT_EQ(i8av_ptr[5],-64);
    
    // using uint16* interface
    conduit_node_set_external_int16_ptr(n,i16av,6);
    conduit_node_print(n);
    
    i16av_ptr = conduit_node_as_int16_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i16av_ptr[i],i16av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i16av_ptr[i],&i16av[i]);
    }
    EXPECT_EQ(i16av_ptr[5],-64);
    
    // using uint32 * interface
    conduit_node_set_external_int32_ptr(n,i32av,6);
    conduit_node_print(n);

    i32av_ptr = conduit_node_as_int32_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i32av_ptr[i],i32av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i32av_ptr[i],&i32av[i]);
    }
    EXPECT_EQ(i32av_ptr[5],-64);
    
    // using uint64 * interface
    conduit_node_set_external_int64_ptr(n,i64av,6);
    conduit_node_print(n);
    
    i64av_ptr = conduit_node_as_int64_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i64av_ptr[i],i64av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i64av_ptr[i],&i64av[i]);
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
    
    //--------------
    // set
    //--------------
    
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

    //--------------
    // set_external
    //--------------

    // using uint8* interface
    conduit_node_set_external_uint8_ptr(n,u8av,6);
    conduit_node_print(n);

    u8av_ptr = conduit_node_as_uint8_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u8av_ptr[i],u8av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u8av_ptr[i],&u8av[i]);
    }
    EXPECT_EQ(u8av_ptr[5],64);
    
    // using uint16* interface
    conduit_node_set_external_uint16_ptr(n,u16av,6);
    conduit_node_print(n);
    
    u16av_ptr = conduit_node_as_uint16_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u16av_ptr[i],u16av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u16av_ptr[i],&u16av[i]);
    }
    EXPECT_EQ(u16av_ptr[5],64);
    
    // using uint32 * interface
    conduit_node_set_external_uint32_ptr(n,u32av,6);
    conduit_node_print(n);

    u32av_ptr = conduit_node_as_uint32_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u32av_ptr[i],u32av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u32av_ptr[i],&u32av[i]);
    }
    EXPECT_EQ(u32av_ptr[5],64);
    
    // using uint64 * interface
    conduit_node_set_external_uint64_ptr(n,u64av,6);
    conduit_node_print(n);
    
    u64av_ptr = conduit_node_as_uint64_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u64av_ptr[i],u64av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u64av_ptr[i],&u64av[i]);
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
    
    //--------------
    // set
    //--------------
    
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

    //--------------
    // set_external
    //--------------

    // float32
    conduit_node_set_external_float32_ptr(n,f32av,4);
    conduit_node_print(n);

    f32av_ptr = conduit_node_as_float32_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f32av_ptr[i],f32av[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f32av_ptr[i],&f32av[i]); 
    }
    EXPECT_NEAR(f32av_ptr[3],-6.4,0.001);
    
    // float32 detailed
    conduit_node_set_external_float32_ptr_detailed(n,
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
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f32av_ptr[i],&f32av[i]); 
    }
    EXPECT_NEAR(f32av_ptr[3],-6.4,0.001);
    
    // float64
    conduit_node_set_external_float64_ptr(n,f64av,4);
    conduit_node_print(n);

    f64av_ptr = conduit_node_as_float64_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f64av_ptr[i],f64av[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);

    // float64 detailed
    conduit_node_set_external_float64_ptr_detailed(n,
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
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
// set_path  and set_path_external tests
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_path_bitwidth_int_ptr)
{
    conduit_int8    i8av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int16  i16av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int32  i32av[6] = {-2,-4,-8,-16,-32,-64};
    conduit_int64  i64av[6] = {-2,-4,-8,-16,-32,-64};
        
    conduit_node *n = conduit_node_create();
    
    //--------------
    // set
    //--------------
    
    // using uint8* interface
    conduit_node_set_path_int8_ptr(n,"i8",i8av,6);
    conduit_node_print(n);

    conduit_int8 *i8av_ptr = conduit_node_fetch_path_as_int8_ptr(n,"i8");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i8av_ptr[i],i8av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i8av_ptr[i],&i8av[i]);
    }
    EXPECT_EQ(i8av_ptr[5],-64);
    
    // using uint16* interface
    conduit_node_set_path_int16_ptr(n,"i16",i16av,6);
    conduit_node_print(n);
    
    conduit_int16 *i16av_ptr = conduit_node_fetch_path_as_int16_ptr(n,"i16");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i16av_ptr[i],i16av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i16av_ptr[i],&i16av[i]);
    }
    EXPECT_EQ(i16av_ptr[5],-64);
    
    // using uint32 * interface
    conduit_node_set_path_int32_ptr(n,"i32",i32av,6);
    conduit_node_print(n);

    conduit_int32 *i32av_ptr = conduit_node_fetch_path_as_int32_ptr(n,"i32");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i32av_ptr[i],i32av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i32av_ptr[i],&i32av[i]);
    }
    EXPECT_EQ(i32av_ptr[5],-64);
    
    // using uint64 * interface
    conduit_node_set_path_int64_ptr(n,"i64",i64av,6);
    conduit_node_print(n);
    
    conduit_int64 *i64av_ptr = conduit_node_fetch_path_as_int64_ptr(n,"i64");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i64av_ptr[i],i64av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&i64av_ptr[i],&i64av[i]);
    }
    EXPECT_EQ(i64av_ptr[5],-64);
    
    //--------------
    // set_external
    //--------------
    
    // using uint8* interface
    conduit_node_set_path_external_int8_ptr(n,"i8",i8av,6);
    conduit_node_print(n);

    i8av_ptr = conduit_node_fetch_path_as_int8_ptr(n,"i8");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i8av_ptr[i],i8av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i8av_ptr[i],&i8av[i]);
    }
    EXPECT_EQ(i8av_ptr[5],-64);
    
    // using uint16* interface
    conduit_node_set_path_external_int16_ptr(n,"i16",i16av,6);
    conduit_node_print(n);
    
    i16av_ptr = conduit_node_fetch_path_as_int16_ptr(n,"i16");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i16av_ptr[i],i16av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i16av_ptr[i],&i16av[i]);
    }
    EXPECT_EQ(i16av_ptr[5],-64);
    
    // using uint32 * interface
    conduit_node_set_path_external_int32_ptr(n,"i32",i32av,6);
    conduit_node_print(n);

    i32av_ptr = conduit_node_fetch_path_as_int32_ptr(n,"i32");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i32av_ptr[i],i32av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i32av_ptr[i],&i32av[i]);
    }
    EXPECT_EQ(i32av_ptr[5],-64);
    
    // using uint64 * interface
    conduit_node_set_path_external_int64_ptr(n,"i64",i64av,6);
    conduit_node_print(n);
    
    i64av_ptr = conduit_node_fetch_path_as_int64_ptr(n,"i64");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(i64av_ptr[i],i64av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&i64av_ptr[i],&i64av[i]);
    }
    EXPECT_EQ(i64av_ptr[5],-64);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_path_bitwidth_uint_ptr)
{
    conduit_uint8    u8av[6] = {2,4,8,16,32,64};
    conduit_uint16  u16av[6] = {2,4,8,16,32,64};
    conduit_uint32  u32av[6] = {2,4,8,16,32,64};
    conduit_uint64  u64av[6] = {2,4,8,16,32,64};
        
    conduit_node *n = conduit_node_create();
    
    //--------------
    // set
    //--------------
    
    // using uint8* interface
    conduit_node_set_path_uint8_ptr(n,"u8",u8av,6);
    conduit_node_print(n);

    conduit_uint8 *u8av_ptr = conduit_node_fetch_path_as_uint8_ptr(n,"u8");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u8av_ptr[i],u8av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u8av_ptr[i],&u8av[i]);
    }
    EXPECT_EQ(u8av_ptr[5],64);
    
    // using uint16* interface
    conduit_node_set_path_uint16_ptr(n,"u16",u16av,6);
    conduit_node_print(n);
    
    conduit_uint16 *u16av_ptr = conduit_node_fetch_path_as_uint16_ptr(n,"u16");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u16av_ptr[i],u16av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u16av_ptr[i],&u16av[i]);
    }
    EXPECT_EQ(u16av_ptr[5],64);
    
    // using uint32 * interface
    conduit_node_set_path_uint32_ptr(n,"u32",u32av,6);
    conduit_node_print(n);

    conduit_uint32 *u32av_ptr = conduit_node_fetch_path_as_uint32_ptr(n,"u32");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u32av_ptr[i],u32av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u32av_ptr[i],&u32av[i]);
    }
    EXPECT_EQ(u32av_ptr[5],64);
    
    // using uint64 * interface
    conduit_node_set_path_uint64_ptr(n,"u64",u64av,6);
    conduit_node_print(n);
    
    conduit_uint64 *u64av_ptr = conduit_node_fetch_path_as_uint64_ptr(n,"u64");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u64av_ptr[i],u64av[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&u64av_ptr[i],&u64av[i]);
    }
    EXPECT_EQ(u64av_ptr[5],64);

    //--------------
    // set_external
    //--------------

    // using uint8* interface
    conduit_node_set_path_external_uint8_ptr(n,"u8",u8av,6);
    conduit_node_print(n);

    u8av_ptr = conduit_node_fetch_path_as_uint8_ptr(n,"u8");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u8av_ptr[i],u8av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u8av_ptr[i],&u8av[i]);
    }
    EXPECT_EQ(u8av_ptr[5],64);
    
    // using uint16* interface
    conduit_node_set_path_external_uint16_ptr(n,"u16",u16av,6);
    conduit_node_print(n);
    
    u16av_ptr = conduit_node_fetch_path_as_uint16_ptr(n,"u16");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u16av_ptr[i],u16av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u16av_ptr[i],&u16av[i]);
    }
    EXPECT_EQ(u16av_ptr[5],64);
    
    // using uint32 * interface
    conduit_node_set_path_external_uint32_ptr(n,"u32",u32av,6);
    conduit_node_print(n);

    u32av_ptr = conduit_node_fetch_path_as_uint32_ptr(n,"u32");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u32av_ptr[i],u32av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u32av_ptr[i],&u32av[i]);
    }
    EXPECT_EQ(u32av_ptr[5],64);
    
    // using uint64 * interface
    conduit_node_set_path_external_uint64_ptr(n,"u64",u64av,6);
    conduit_node_print(n);
    
    u64av_ptr = conduit_node_fetch_path_as_uint64_ptr(n,"u64");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(u64av_ptr[i],u64av[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&u64av_ptr[i],&u64av[i]);
    }
    EXPECT_EQ(u64av_ptr[5],64);

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_set, set_path_bitwidth_float_ptr)
{
    conduit_float32  f32av[4] = {-0.8f, -1.6f, -3.2f, -6.4f};
    conduit_float64  f64av[4] = {-0.8, -1.6, -3.2, -6.4};

    conduit_node *n = conduit_node_create();
    
    //--------------
    // set
    //--------------
    
    // float32
    conduit_node_set_path_float32_ptr(n,"f32",f32av,4);
    conduit_node_print(n);

    conduit_float32 *f32av_ptr = conduit_node_fetch_path_as_float32_ptr(n,"f32");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f32av_ptr[i],f32av[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&f32av_ptr[i],&f32av[i]); 
    }
    EXPECT_NEAR(f32av_ptr[3],-6.4,0.001);
    
    // float32 detailed
    conduit_node_set_path_float32_ptr_detailed(n,
                                               "f32",
                                               f32av,
                                               4,
                                               0,
                                               sizeof(conduit_float32),
                                               sizeof(conduit_float32),
                                               CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    f32av_ptr = conduit_node_fetch_path_as_float32_ptr(n,"f32");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f32av_ptr[i],f32av[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&f32av_ptr[i],&f32av[i]); 
    }
    EXPECT_NEAR(f32av_ptr[3],-6.4,0.001);
    
    // float64
    conduit_node_set_path_float64_ptr(n,"f64",f64av,4);
    conduit_node_print(n);

    conduit_float64 *f64av_ptr = conduit_node_fetch_path_as_float64_ptr(n,"f64");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f64av_ptr[i],f64av[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);

    // float64 detailed
    conduit_node_set_path_float64_ptr_detailed(n,
                                               "f64",
                                               f64av,
                                               4,
                                               0,
                                               sizeof(conduit_float64),
                                               sizeof(conduit_float64),
                                               CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    f64av_ptr = conduit_node_fetch_path_as_float64_ptr(n,"f64");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f64av_ptr[i],f64av[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);

    //--------------
    // set_external
    //--------------

    // float32
    conduit_node_set_path_external_float32_ptr(n,"f32",f32av,4);
    conduit_node_print(n);

    f32av_ptr = conduit_node_fetch_path_as_float32_ptr(n,"f32");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f32av_ptr[i],f32av[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f32av_ptr[i],&f32av[i]); 
    }
    EXPECT_NEAR(f32av_ptr[3],-6.4,0.001);
    
    // float32 detailed
    conduit_node_set_path_external_float32_ptr_detailed(n,
                                                        "f32",
                                                        f32av,
                                                        4,
                                                        0,
                                                        sizeof(conduit_float32),
                                                        sizeof(conduit_float32),
                                                        CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    f32av_ptr = conduit_node_fetch_path_as_float32_ptr(n,"f32");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f32av_ptr[i],f32av[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f32av_ptr[i],&f32av[i]); 
    }
    EXPECT_NEAR(f32av_ptr[3],-6.4,0.001);
    
    // float64
    conduit_node_set_path_external_float64_ptr(n,"f64",f64av,4);
    conduit_node_print(n);

    f64av_ptr = conduit_node_fetch_path_as_float64_ptr(n,"f64");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f64av_ptr[i],f64av[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);

    // float64 detailed
    conduit_node_set_path_external_float64_ptr_detailed(n,
                                                        "f64",
                                                        f64av,
                                                        4,
                                                        0,
                                                        sizeof(conduit_float64),
                                                        sizeof(conduit_float64),
                                                        CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    f64av_ptr = conduit_node_fetch_path_as_float64_ptr(n,"f64");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(f64av_ptr[i],f64av[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&f64av_ptr[i],&f64av[i]);
    }
    EXPECT_NEAR(f64av_ptr[3],-6.4,0.001);

    conduit_node_destroy(n);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// c style tests
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// set and set_path tests
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_int_scalar)
{
    char        icv  = 8;
    short       isv  = 16;
    int         iiv  = 32;
    long        ilv  = 64;

    conduit_node *n = conduit_node_create();
    
    //----------------
    // set
    //----------------
    
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
    
    //----------------
    // set_path
    //----------------
    
    // char
    conduit_node_set_path_char(n,"c",icv);
    EXPECT_EQ(conduit_node_fetch_path_as_char(n,"c"),icv);
    conduit_node_print(n);

    // short
    conduit_node_set_path_short(n,"s",isv);
    EXPECT_EQ(conduit_node_fetch_path_as_short(n,"s"),isv);
    conduit_node_print(n);

    // int
    conduit_node_set_path_int(n,"i",iiv);
    EXPECT_EQ(conduit_node_fetch_path_as_int(n,"i"),iiv);
    conduit_node_print(n);
    
    // long
    conduit_node_set_path_long(n,"l",ilv);
    EXPECT_EQ(conduit_node_fetch_path_as_long(n,"l"),ilv);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_signed_int_scalar)
{
    signed char  siscv = -8;
    signed short sisv  = -16;
    signed int   siiv  = -32;
    signed long  silv  = -64;

    conduit_node *n = conduit_node_create();

    //----------------
    // set
    //----------------
    
    // signed char
    conduit_node_set_signed_char(n,siscv);
    EXPECT_EQ(conduit_node_as_signed_char(n),siscv);
    conduit_node_print(n);
    
    // short
    conduit_node_set_signed_short(n,sisv);
    EXPECT_EQ(conduit_node_as_signed_short(n),sisv);
    conduit_node_print(n);

    
    // int
    conduit_node_set_signed_int(n,siiv);
    EXPECT_EQ(conduit_node_as_signed_int(n),siiv);
    conduit_node_print(n);
    
    // signed long
    conduit_node_set_signed_long(n,silv);
    EXPECT_EQ(conduit_node_as_signed_long(n),silv);
    conduit_node_print(n);
    
    //----------------
    // set_path
    //----------------
    
    // signed char
    conduit_node_set_path_signed_char(n,"sc",siscv);
    EXPECT_EQ(conduit_node_fetch_path_as_signed_char(n,"sc"),siscv);
    conduit_node_print(n);
    
    // short
    conduit_node_set_path_signed_short(n,"ss",sisv);
    EXPECT_EQ(conduit_node_fetch_path_as_signed_short(n,"ss"),sisv);
    conduit_node_print(n);

    
    // int
    conduit_node_set_path_signed_int(n,"si",siiv);
    EXPECT_EQ(conduit_node_fetch_path_as_signed_int(n,"si"),siiv);
    conduit_node_print(n);
    
    // signed long
    conduit_node_set_path_signed_long(n,"sl",silv);
    EXPECT_EQ(conduit_node_fetch_path_as_signed_long(n,"sl"),silv);
    conduit_node_print(n);
    
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_unsigned_int_scalar)
{
    unsigned char   ucv = 8;
    unsigned short  usv = 16;
    unsigned int    uiv = 32;
    unsigned long   ulv = 64;

    conduit_node *n = conduit_node_create();
    
    //----------------
    // set
    //----------------

    // unsigned char
    conduit_node_set_unsigned_char(n,ucv);
    EXPECT_EQ(conduit_node_as_unsigned_char(n),ucv);
    conduit_node_print(n);
    
    // unsigned short
    conduit_node_set_unsigned_short(n,usv);
    EXPECT_EQ(conduit_node_as_unsigned_short(n),usv);
    conduit_node_print(n);

    // unsigned int
    conduit_node_set_unsigned_int(n,uiv);
    EXPECT_EQ(conduit_node_as_unsigned_int(n),uiv);
    conduit_node_print(n);
    
    // unsigned long
    conduit_node_set_unsigned_long(n,ulv);
    EXPECT_EQ(conduit_node_as_unsigned_long(n),ulv);
    conduit_node_print(n);
    
    //----------------
    // set_path
    //----------------
    
    // unsigned char
    conduit_node_set_path_unsigned_char(n,"uc",ucv);
    EXPECT_EQ(conduit_node_fetch_path_as_unsigned_char(n,"uc"),ucv);
    conduit_node_print(n);
    
    // unsigned short
    conduit_node_set_path_unsigned_short(n,"us",usv);
    EXPECT_EQ(conduit_node_fetch_path_as_unsigned_short(n,"us"),usv);
    conduit_node_print(n);

    // unsigned int
    conduit_node_set_path_unsigned_int(n,"ui",uiv);
    EXPECT_EQ(conduit_node_fetch_path_as_unsigned_int(n,"ui"),uiv);
    conduit_node_print(n);
    
    // unsigned long
    conduit_node_set_path_unsigned_long(n,"ul",ulv);
    EXPECT_EQ(conduit_node_fetch_path_as_unsigned_long(n,"ul"),ulv);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_float_scalar)
{
    float  fv =  3.1415f;
    double dv = -3.1415;

    conduit_node *n = conduit_node_create();

    //----------------
    // set
    //----------------

    // float32
    conduit_node_set_float(n,fv);
    EXPECT_EQ(conduit_node_as_float(n),fv);
    conduit_node_print(n);
    
    // float64
    conduit_node_set_double(n,dv);
    EXPECT_EQ(conduit_node_as_double(n),dv);
    conduit_node_print(n);
    
    //----------------
    // set path
    //----------------
    
    // float32
    conduit_node_set_path_float(n,"f",fv);
    EXPECT_EQ(conduit_node_fetch_path_as_float(n,"f"),fv);
    conduit_node_print(n);
    
    // float64
    conduit_node_set_path_double(n,"d",dv);
    EXPECT_EQ(conduit_node_fetch_path_as_double(n,"d"),dv);
    conduit_node_print(n);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
// set and set_external tests
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_int_ptr)
{
    char  icav[6]  = {2,4,8,16,32,64};
    short isav[6]  = {2,4,8,16,32,64};
    int   iiav[6]  = {2,4,8,16,32,64};
    long  ilav[6]  = {2,4,8,16,32,64};
        
    conduit_node *n = conduit_node_create();

    //----------------
    // set 
    //----------------
    
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
    EXPECT_EQ(icav_ptr[5],64);
    
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
    EXPECT_EQ(isav_ptr[5],64);
    
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
    EXPECT_EQ(iiav_ptr[5],64);
    
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
    EXPECT_EQ(ilav_ptr[5],64);

    //----------------
    // set_external
    //----------------
    
    // using char* interface
    conduit_node_set_external_char_ptr(n,icav,6);
    conduit_node_print(n);

    icav_ptr = conduit_node_as_char_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(icav_ptr[i],icav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&icav_ptr[i],&icav[i]);
    }
    EXPECT_EQ(icav_ptr[5],64);
    
    // using short* interface
    conduit_node_set_external_short_ptr(n,isav,6);
    conduit_node_print(n);
    
    isav_ptr = conduit_node_as_short_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(isav_ptr[i],isav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&isav_ptr[i],&isav[i]);
    }
    EXPECT_EQ(isav_ptr[5],64);
    
    // using int* interface
    conduit_node_set_external_int_ptr(n,iiav,6);
    conduit_node_print(n);

    iiav_ptr = conduit_node_as_int_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(iiav_ptr[i],iiav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&iiav_ptr[i],&iiav[i]);
    }
    EXPECT_EQ(iiav_ptr[5],64);
    
    // using long * interface
    conduit_node_set_external_long_ptr(n,ilav,6);
    conduit_node_print(n);
    
    ilav_ptr = conduit_node_as_long_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ilav_ptr[i],ilav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&ilav_ptr[i],&ilav[i]);
    }
    EXPECT_EQ(ilav_ptr[5],64);
    
    
    conduit_node_destroy(n);

}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_signed_int_ptr)
{
    signed char  sicav[6]  = {-2,-4,-8,-16,-32,-64};
    signed short sisav[6]  = {-2,-4,-8,-16,-32,-64};
    signed int   siiav[6]  = {-2,-4,-8,-16,-32,-64};
    signed long  silav[6]  = {-2,-4,-8,-16,-32,-64};
        
    conduit_node *n = conduit_node_create();

    //----------------
    // set
    //----------------
    
    // using signed char* interface
    conduit_node_set_signed_char_ptr(n,sicav,6);
    conduit_node_print(n);

    signed char *sicav_ptr = conduit_node_as_signed_char_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sicav_ptr[i],sicav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&sicav_ptr[i],&sicav[i]);
    }
    EXPECT_EQ(sicav_ptr[5],-64);
    
    // using signed short* interface
    conduit_node_set_signed_short_ptr(n,sisav,6);
    conduit_node_print(n);
    
    signed short *sisav_ptr = conduit_node_as_signed_short_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sisav_ptr[i],sisav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&sisav_ptr[i],&sisav[i]);
    }
    EXPECT_EQ(sisav_ptr[5],-64);
    
    // using int* interface
    conduit_node_set_signed_int_ptr(n,siiav,6);
    conduit_node_print(n);

    signed int *siiav_ptr = conduit_node_as_signed_int_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(siiav_ptr[i],siiav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&siiav_ptr[i],&siiav[i]);
    }
    EXPECT_EQ(siiav_ptr[5],-64);
    
    // using signed long * interface
    conduit_node_set_signed_long_ptr(n,silav,6);
    conduit_node_print(n);
    
    signed long *silav_ptr = conduit_node_as_signed_long_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(silav_ptr[i],silav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&silav_ptr[i],&silav[i]);
    }
    EXPECT_EQ(silav_ptr[5],-64);
    
    //----------------
    // set_external
    //----------------
    
    // using signed char* interface
    conduit_node_set_external_signed_char_ptr(n,sicav,6);
    conduit_node_print(n);

    sicav_ptr = conduit_node_as_signed_char_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sicav_ptr[i],sicav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&sicav_ptr[i],&sicav[i]);
    }
    EXPECT_EQ(sicav_ptr[5],-64);
    
    // using signed short* interface
    conduit_node_set_external_signed_short_ptr(n,sisav,6);
    conduit_node_print(n);
    
    sisav_ptr = conduit_node_as_signed_short_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sisav_ptr[i],sisav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&sisav_ptr[i],&sisav[i]);
    }
    EXPECT_EQ(sisav_ptr[5],-64);
    
    // using int* interface
    conduit_node_set_external_signed_int_ptr(n,siiav,6);
    conduit_node_print(n);

    siiav_ptr = conduit_node_as_signed_int_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(siiav_ptr[i],siiav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&siiav_ptr[i],&siiav[i]);
    }
    EXPECT_EQ(siiav_ptr[5],-64);
    
    // using signed long * interface
    conduit_node_set_external_signed_long_ptr(n,silav,6);
    conduit_node_print(n);
    
    silav_ptr = conduit_node_as_signed_long_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(silav_ptr[i],silav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&silav_ptr[i],&silav[i]);
    }
    EXPECT_EQ(silav_ptr[5],-64);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_native_unsigned_int_ptr)
{
    unsigned char   ucav[6] = {2,4,8,16,32,64};
    unsigned short  usav[6] = {2,4,8,16,32,64};
    unsigned int    uiav[6] = {2,4,8,16,32,64};
    unsigned long   ulav[6] = {2,4,8,16,32,64};
        
    conduit_node *n = conduit_node_create();

    //----------------
    // set
    //----------------

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
    
    //----------------
    // set_external
    //----------------
    
    // using unsigned char* interface
    conduit_node_set_external_unsigned_char_ptr(n,ucav,6);
    conduit_node_print(n);
    
    ucav_ptr = conduit_node_as_unsigned_char_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ucav_ptr[i],ucav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&ucav_ptr[i],&ucav[i]);
    }
    EXPECT_EQ(ucav_ptr[5],64);
    
    // using unsigned short* interface
    conduit_node_set_external_unsigned_short_ptr(n,usav,6);
    conduit_node_print(n);
    
    usav_ptr = conduit_node_as_unsigned_short_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(usav_ptr[i],usav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&usav_ptr[i],&usav[i]);
    }
    EXPECT_EQ(usav_ptr[5],64);
    
    // using unsigned int * interface
    conduit_node_set_external_unsigned_int_ptr(n,uiav,6);
    conduit_node_print(n);

    uiav_ptr = conduit_node_as_unsigned_int_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(uiav_ptr[i],uiav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&uiav_ptr[i],&uiav[i]);
    }
    EXPECT_EQ(uiav_ptr[5],64);
    
    // using unsigned long * interface
    conduit_node_set_external_unsigned_long_ptr(n,ulav,6);
    conduit_node_print(n);
    
    ulav_ptr = conduit_node_as_unsigned_long_ptr(n);
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ulav_ptr[i],ulav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&ulav_ptr[i],&ulav[i]);
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

    //----------------
    // set 
    //----------------
    
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

    //----------------
    // set_external
    //----------------
    
    // float*
    conduit_node_set_external_float_ptr(n,fav,4);
    conduit_node_print(n);

    fav_ptr = conduit_node_as_float_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(fav_ptr[i],fav[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // float32 detailed
    conduit_node_set_external_float_ptr_detailed(n,
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
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // double
    conduit_node_set_external_double_ptr(n,dav,4);
    conduit_node_print(n);

    dav_ptr = conduit_node_as_double_ptr(n);
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(dav_ptr[i],dav[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);

    // float64 detailed
    conduit_node_set_external_double_ptr_detailed(n,
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
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);


    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
// set_path and set_path_external tests
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_path_native_int_ptr)
{
    char  icav[6]  = {2,4,8,16,32,64};
    short isav[6]  = {2,4,8,16,32,64};
    int   iiav[6]  = {2,4,8,16,32,64};
    long  ilav[6]  = {2,4,8,16,32,64};
        
    conduit_node *n = conduit_node_create();

    //----------------
    // set 
    //----------------
    
    // using char* interface
    conduit_node_set_path_char_ptr(n,"c",icav,6);
    conduit_node_print(n);

    char *icav_ptr = conduit_node_fetch_path_as_char_ptr(n,"c");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(icav_ptr[i],icav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&icav_ptr[i],&icav[i]);
    }
    EXPECT_EQ(icav_ptr[5],64);
    
    // using short* interface
    conduit_node_set_path_short_ptr(n,"s",isav,6);
    conduit_node_print(n);
    
    short *isav_ptr = conduit_node_fetch_path_as_short_ptr(n,"s");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(isav_ptr[i],isav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&isav_ptr[i],&isav[i]);
    }
    EXPECT_EQ(isav_ptr[5],64);
    
    // using int* interface
    conduit_node_set_path_int_ptr(n,"i",iiav,6);
    conduit_node_print(n);

    int *iiav_ptr = conduit_node_fetch_path_as_int_ptr(n,"i");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(iiav_ptr[i],iiav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&iiav_ptr[i],&iiav[i]);
    }
    EXPECT_EQ(iiav_ptr[5],64);
    
    // using long * interface
    conduit_node_set_path_long_ptr(n,"l",ilav,6);
    conduit_node_print(n);
    
    long *ilav_ptr = conduit_node_fetch_path_as_long_ptr(n,"l");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ilav_ptr[i],ilav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&ilav_ptr[i],&ilav[i]);
    }
    EXPECT_EQ(ilav_ptr[5],64);

    //----------------
    // set_external
    //----------------
    
    // using char* interface
    conduit_node_set_path_external_char_ptr(n,"c",icav,6);
    conduit_node_print(n);

    icav_ptr = conduit_node_fetch_path_as_char_ptr(n,"c");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(icav_ptr[i],icav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&icav_ptr[i],&icav[i]);
    }
    EXPECT_EQ(icav_ptr[5],64);
    
    // using short* interface
    conduit_node_set_path_external_short_ptr(n,"s",isav,6);
    conduit_node_print(n);
    
    isav_ptr = conduit_node_fetch_path_as_short_ptr(n,"s");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(isav_ptr[i],isav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&isav_ptr[i],&isav[i]);
    }
    EXPECT_EQ(isav_ptr[5],64);
    
    // using int* interface
    conduit_node_set_path_external_int_ptr(n,"i",iiav,6);
    conduit_node_print(n);

    iiav_ptr = conduit_node_fetch_path_as_int_ptr(n,"i");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(iiav_ptr[i],iiav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&iiav_ptr[i],&iiav[i]);
    }
    EXPECT_EQ(iiav_ptr[5],64);
    
    // using long * interface
    conduit_node_set_path_external_long_ptr(n,"l",ilav,6);
    conduit_node_print(n);
    
    ilav_ptr = conduit_node_fetch_path_as_long_ptr(n,"l");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ilav_ptr[i],ilav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&ilav_ptr[i],&ilav[i]);
    }
    EXPECT_EQ(ilav_ptr[5],64);
    
    
    conduit_node_destroy(n);

}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_path_native_signed_int_ptr)
{
    signed char  sicav[6]  = {-2,-4,-8,-16,-32,-64};
    signed short sisav[6]  = {-2,-4,-8,-16,-32,-64};
    signed int   siiav[6]  = {-2,-4,-8,-16,-32,-64};
    signed long  silav[6]  = {-2,-4,-8,-16,-32,-64};
        
    conduit_node *n = conduit_node_create();

    //----------------
    // set
    //----------------
    
    // using signed char* interface
    conduit_node_set_path_signed_char_ptr(n,"sc",sicav,6);
    conduit_node_print(n);

    signed char *sicav_ptr = conduit_node_fetch_path_as_signed_char_ptr(n,"sc");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sicav_ptr[i],sicav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&sicav_ptr[i],&sicav[i]);
    }
    EXPECT_EQ(sicav_ptr[5],-64);
    
    // using signed short* interface
    conduit_node_set_path_signed_short_ptr(n,"ss",sisav,6);
    conduit_node_print(n);
    
    signed short *sisav_ptr = conduit_node_fetch_path_as_signed_short_ptr(n,"ss");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sisav_ptr[i],sisav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&sisav_ptr[i],&sisav[i]);
    }
    EXPECT_EQ(sisav_ptr[5],-64);
    
    // using int* interface
    conduit_node_set_path_signed_int_ptr(n,"si",siiav,6);
    conduit_node_print(n);

    signed int *siiav_ptr = conduit_node_fetch_path_as_signed_int_ptr(n,"si");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(siiav_ptr[i],siiav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&siiav_ptr[i],&siiav[i]);
    }
    EXPECT_EQ(siiav_ptr[5],-64);
    
    // using signed long * interface
    conduit_node_set_path_signed_long_ptr(n,"sl",silav,6);
    conduit_node_print(n);
    
    signed long *silav_ptr = conduit_node_fetch_path_as_signed_long_ptr(n,"sl");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(silav_ptr[i],silav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&silav_ptr[i],&silav[i]);
    }
    EXPECT_EQ(silav_ptr[5],-64);
    
    //----------------
    // set_external
    //----------------
    
    // using signed char* interface
    conduit_node_set_path_external_signed_char_ptr(n,"sc",sicav,6);
    conduit_node_print(n);

    sicav_ptr = conduit_node_fetch_path_as_signed_char_ptr(n,"sc");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sicav_ptr[i],sicav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&sicav_ptr[i],&sicav[i]);
    }
    EXPECT_EQ(sicav_ptr[5],-64);
    
    // using signed short* interface
    conduit_node_set_path_external_signed_short_ptr(n,"ss",sisav,6);
    conduit_node_print(n);
    
    sisav_ptr = conduit_node_fetch_path_as_signed_short_ptr(n,"ss");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(sisav_ptr[i],sisav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&sisav_ptr[i],&sisav[i]);
    }
    EXPECT_EQ(sisav_ptr[5],-64);
    
    // using int* interface
    conduit_node_set_path_external_signed_int_ptr(n,"si",siiav,6);
    conduit_node_print(n);

    siiav_ptr = conduit_node_fetch_path_as_signed_int_ptr(n,"si");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(siiav_ptr[i],siiav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&siiav_ptr[i],&siiav[i]);
    }
    EXPECT_EQ(siiav_ptr[5],-64);
    
    // using signed long * interface
    conduit_node_set_path_external_signed_long_ptr(n,"sl",silav,6);
    conduit_node_print(n);
    
    silav_ptr = conduit_node_fetch_path_as_signed_long_ptr(n,"sl");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(silav_ptr[i],silav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&silav_ptr[i],&silav[i]);
    }
    EXPECT_EQ(silav_ptr[5],-64);
    
    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(c_conduit_node_set, set_path_native_unsigned_int_ptr)
{
    unsigned char   ucav[6] = {2,4,8,16,32,64};
    unsigned short  usav[6] = {2,4,8,16,32,64};
    unsigned int    uiav[6] = {2,4,8,16,32,64};
    unsigned long   ulav[6] = {2,4,8,16,32,64};
        
    conduit_node *n = conduit_node_create();

    //----------------
    // set
    //----------------

    // using unsigned char* interface
    conduit_node_set_path_unsigned_char_ptr(n,"uc",ucav,6);
    conduit_node_print(n);

    unsigned char *ucav_ptr = conduit_node_fetch_path_as_unsigned_char_ptr(n,"uc");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ucav_ptr[i],ucav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&ucav_ptr[i],&ucav[i]);
    }
    EXPECT_EQ(ucav_ptr[5],64);
    
    // using unsigned short* interface
    conduit_node_set_path_unsigned_short_ptr(n,"us",usav,6);
    conduit_node_print(n);
    
    unsigned short *usav_ptr = conduit_node_fetch_path_as_unsigned_short_ptr(n,"us");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(usav_ptr[i],usav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&usav_ptr[i],&usav[i]);
    }
    EXPECT_EQ(usav_ptr[5],64);
    
    // using unsigned int * interface
    conduit_node_set_path_unsigned_int_ptr(n,"ui",uiav,6);
    conduit_node_print(n);

    unsigned int *uiav_ptr = conduit_node_fetch_path_as_unsigned_int_ptr(n,"ui");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(uiav_ptr[i],uiav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&uiav_ptr[i],&uiav[i]);
    }
    EXPECT_EQ(uiav_ptr[5],64);
    
    // using unsigned long * interface
    conduit_node_set_path_unsigned_long_ptr(n,"ul",ulav,6);
    conduit_node_print(n);
    
    unsigned long *ulav_ptr = conduit_node_fetch_path_as_unsigned_long_ptr(n,"ul");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ulav_ptr[i],ulav[i]);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&ulav_ptr[i],&ulav[i]);
    }
    EXPECT_EQ(ulav_ptr[5],64);
    
    //----------------
    // set_external
    //----------------
    
    // using unsigned char* interface
    conduit_node_set_path_external_unsigned_char_ptr(n,"uc",ucav,6);
    conduit_node_print(n);
    
    ucav_ptr = conduit_node_fetch_path_as_unsigned_char_ptr(n,"uc");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ucav_ptr[i],ucav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&ucav_ptr[i],&ucav[i]);
    }
    EXPECT_EQ(ucav_ptr[5],64);
    
    // using unsigned short* interface
    conduit_node_set_path_external_unsigned_short_ptr(n,"us",usav,6);
    conduit_node_print(n);
    
    usav_ptr = conduit_node_fetch_path_as_unsigned_short_ptr(n,"us");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(usav_ptr[i],usav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&usav_ptr[i],&usav[i]);
    }
    EXPECT_EQ(usav_ptr[5],64);
    
    // using unsigned int * interface
    conduit_node_set_path_external_unsigned_int_ptr(n,"ui",uiav,6);
    conduit_node_print(n);

    uiav_ptr = conduit_node_fetch_path_as_unsigned_int_ptr(n,"ui");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(uiav_ptr[i],uiav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&uiav_ptr[i],&uiav[i]);
    }
    EXPECT_EQ(uiav_ptr[5],64);
    
    // using unsigned long * interface
    conduit_node_set_path_external_unsigned_long_ptr(n,"ul",ulav,6);
    conduit_node_print(n);
    
    ulav_ptr = conduit_node_fetch_path_as_unsigned_long_ptr(n,"ul");
    for(conduit_index_t i=0;i<6;i++)
    {
        EXPECT_EQ(ulav_ptr[i],ulav[i]);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&ulav_ptr[i],&ulav[i]);
    }
    EXPECT_EQ(ulav_ptr[5],64);

    conduit_node_destroy(n);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_set, set_path_native_float_ptr)
{
    float   fav[4] = {-0.8f, -1.6f, -3.2f, -6.4f};
    double  dav[4] = {-0.8, -1.6, -3.2, -6.4};

    conduit_node *n = conduit_node_create();

    //----------------
    // set 
    //----------------
    
    // float*
    conduit_node_set_path_float_ptr(n,"f",fav,4);
    conduit_node_print(n);

    float *fav_ptr = conduit_node_fetch_path_as_float_ptr(n,"f");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(fav_ptr[i],fav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // float32 detailed
    conduit_node_set_path_float_ptr_detailed(n,
                                             "f",
                                             fav,
                                             4,
                                             0,
                                             sizeof(float),
                                             sizeof(float),
                                             CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    fav_ptr = conduit_node_fetch_path_as_float_ptr(n,"f");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(fav_ptr[i],fav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // double
    conduit_node_set_path_double_ptr(n,"d",dav,4);
    conduit_node_print(n);

    double *dav_ptr = conduit_node_fetch_path_as_double_ptr(n,"d");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(dav_ptr[i],dav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);

    // float64 detailed
    conduit_node_set_path_double_ptr_detailed(n,
                                              "d",
                                              dav,
                                              4,
                                              0,
                                              sizeof(double),
                                              sizeof(double),
                                              CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    dav_ptr = conduit_node_fetch_path_as_double_ptr(n,"d");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(dav_ptr[i],dav[i],0.001);
        // set(...) semantics imply a copy -- mem addys should differ
        EXPECT_NE(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);

    //----------------
    // set_external
    //----------------
    
    // float*
    conduit_node_set_path_external_float_ptr(n,"f",fav,4);
    conduit_node_print(n);

    fav_ptr = conduit_node_fetch_path_as_float_ptr(n,"f");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(fav_ptr[i],fav[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // float32 detailed
    conduit_node_set_path_external_float_ptr_detailed(n,
                                                      "f",
                                                      fav,
                                                      4,
                                                      0,
                                                      sizeof(float),
                                                      sizeof(float),
                                                      CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    fav_ptr = conduit_node_fetch_path_as_float_ptr(n,"f");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(fav_ptr[i],fav[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&fav_ptr[i],&fav[i]); 
    }
    EXPECT_NEAR(fav_ptr[3],-6.4,0.001);
    
    
    // double
    conduit_node_set_path_external_double_ptr(n,"d",dav,4);
    conduit_node_print(n);

    dav_ptr = conduit_node_fetch_path_as_double_ptr(n,"d");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(dav_ptr[i],dav[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);

    // float64 detailed
    conduit_node_set_path_external_double_ptr_detailed(n,
                                                       "d",
                                                       dav,
                                                       4,
                                                       0,
                                                       sizeof(double),
                                                       sizeof(double),
                                                       CONDUIT_ENDIANNESS_DEFAULT_ID);
    conduit_node_print(n);

    dav_ptr = conduit_node_fetch_path_as_double_ptr(n,"d");
    for(conduit_index_t i=0;i<4;i++)
    {
        EXPECT_NEAR(dav_ptr[i],dav[i],0.001);
        // set_external(...) semantics implies zero-copy -- mem addys should equal
        EXPECT_EQ(&dav_ptr[i],&dav[i]);
    }
    EXPECT_NEAR(dav_ptr[3],-6.4,0.001);


    conduit_node_destroy(n);
}
