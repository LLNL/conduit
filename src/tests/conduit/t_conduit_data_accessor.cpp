// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_array.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_data_accessor, value)
{

    Node n;
    n.set((int8)10);

    int8_accessor  i8_acc  = n.value();
    int16_accessor i16_acc = n.value();
    int32_accessor i32_acc = n.value();
    int64_accessor i64_acc = n.value();

    uint8_accessor  ui8_acc  = n.value();
    uint16_accessor ui16_acc = n.value();
    uint32_accessor ui32_acc = n.value();
    uint64_accessor ui64_acc = n.value();
    
    
    float32_accessor f32_acc = n.value();
    float64_accessor f64_acc = n.value();
    
    EXPECT_EQ(i8_acc[0],(int8)(10));
    EXPECT_EQ(i16_acc[0],(int16)(10));
    EXPECT_EQ(i32_acc[0],(int32)(10));
    EXPECT_EQ(i64_acc[0],(int64)(10));
    
    
    EXPECT_EQ(ui8_acc[0],(uint8)(10));
    EXPECT_EQ(ui16_acc[0],(uint16)(10));
    EXPECT_EQ(ui32_acc[0],(uint32)(10));
    EXPECT_EQ(ui64_acc[0],(uint64)(10));

    EXPECT_EQ(f32_acc[0],(float32)(10));
    EXPECT_EQ(f64_acc[0],(float64)(10));

}


//-----------------------------------------------------------------------------
TEST(conduit_data_accessor, as_bitwidth_style)
{

    Node n;
    n.set((int8)10);

    int8_accessor  i8_acc  = n.as_int8_accessor();
    int16_accessor i16_acc = n.as_int16_accessor();
    int32_accessor i32_acc = n.as_int32_accessor();
    int64_accessor i64_acc = n.as_int64_accessor();

    uint8_accessor  ui8_acc  = n.as_uint8_accessor();
    uint16_accessor ui16_acc = n.as_uint16_accessor();
    uint32_accessor ui32_acc = n.as_uint32_accessor();
    uint64_accessor ui64_acc = n.as_uint64_accessor();
    
    
    float32_accessor f32_acc = n.as_float32_accessor();
    float64_accessor f64_acc = n.as_float64_accessor();
    
    EXPECT_EQ(i8_acc[0],(int8)(10));
    EXPECT_EQ(i16_acc[0],(int16)(10));
    EXPECT_EQ(i32_acc[0],(int32)(10));
    EXPECT_EQ(i64_acc[0],(int64)(10));
    
    
    EXPECT_EQ(ui8_acc[0],(uint8)(10));
    EXPECT_EQ(ui16_acc[0],(uint16)(10));
    EXPECT_EQ(ui32_acc[0],(uint32)(10));
    EXPECT_EQ(ui64_acc[0],(uint64)(10));

    EXPECT_EQ(f32_acc[0],(float32)(10));
    EXPECT_EQ(f64_acc[0],(float64)(10));

}


//-----------------------------------------------------------------------------
TEST(conduit_data_accessor, summary_stats)
{
    std::vector<int64>   v_int64 ={-1,0,1};
    std::vector<uint64>  v_uint64 = {1,2,3};
    std::vector<float64> v_float64= {-1.0,0.0,1.0};

    int64_accessor   va_int64(&v_int64[0],DataType::int64(3));
    uint64_accessor  va_uint64(&v_uint64[0],DataType::uint64(3));
    float64_accessor va_float64(&v_float64[0],DataType::float64(3));


    EXPECT_EQ(va_int64.min(),-1);
    EXPECT_EQ(va_int64.max(),1);
    EXPECT_EQ(va_int64.mean(),0);
    EXPECT_EQ(va_int64.sum(),0);
    EXPECT_EQ(va_int64.count(-1),1);

    EXPECT_EQ(va_uint64.min(),1);
    EXPECT_EQ(va_uint64.max(),3);
    EXPECT_EQ(va_uint64.mean(),2);
    EXPECT_EQ(va_uint64.sum(),6);
    EXPECT_EQ(va_uint64.count(2),1);

    EXPECT_EQ(va_float64.min(),-1.0);
    EXPECT_EQ(va_float64.max(),1.0);
    EXPECT_EQ(va_float64.mean(),0.0);
    EXPECT_EQ(va_float64.sum(),0.0);
    EXPECT_EQ(va_float64.count(0.0),1);

}



//-----------------------------------------------------------------------------
TEST(conduit_data_accessor, as_cstyle)
{

    Node n;
    n.set((int8)10);

    char_accessor         c_acc  = n.as_char_accessor();
    signed_char_accessor  sc_acc = n.as_signed_char_accessor();
    signed_short_accessor ss_acc = n.as_signed_short_accessor();
    signed_int_accessor   si_acc = n.as_signed_int_accessor();
    signed_long_accessor  sl_acc = n.as_signed_long_accessor();

#ifdef CONDUIT_HAS_LONG_LONG
    signed_long_long_accessor  sll_acc = n.as_signed_long_long_accessor();
#endif

    unsigned_char_accessor  usc_acc = n.as_unsigned_char_accessor();
    unsigned_short_accessor uss_acc = n.as_unsigned_short_accessor();
    unsigned_int_accessor   usi_acc = n.as_unsigned_int_accessor();
    unsigned_long_accessor  usl_acc = n.as_unsigned_long_accessor();

#ifdef CONDUIT_HAS_LONG_LONG
    unsigned_long_long_accessor  usll_acc = n.as_unsigned_long_long_accessor();
#endif

    float_accessor  f_acc = n.as_float_accessor();
    double_accessor d_acc = n.as_double_accessor();

#ifdef CONDUIT_USE_LONG_DOUBLE
    long_double_accessor  ld_acc = n.as_long_double_accessor();
#endif

    EXPECT_EQ(c_acc[0],(char)(10));
    EXPECT_EQ(sc_acc[0],(signed char)(10));
    EXPECT_EQ(ss_acc[0],(signed short)(10));
    EXPECT_EQ(si_acc[0],(signed int)(10));
    EXPECT_EQ(sl_acc[0],(signed long)(10));

#ifdef CONDUIT_HAS_LONG_LONG
    EXPECT_EQ(sll_acc[0],(signed long long)(10));
#endif

    EXPECT_EQ(usc_acc[0],(unsigned char)(10));
    EXPECT_EQ(uss_acc[0],(unsigned short)(10));
    EXPECT_EQ(usi_acc[0],(unsigned int)(10));
    EXPECT_EQ(usl_acc[0],(unsigned long)(10));

#ifdef CONDUIT_HAS_LONG_LONG
    EXPECT_EQ(usll_acc[0],(unsigned long long)(10));
#endif

    EXPECT_EQ(f_acc[0],(float)(10));
    EXPECT_EQ(d_acc[0],(double)(10));

#ifdef CONDUIT_USE_LONG_DOUBLE
    EXPECT_EQ(ld_acc[0],(long double)(10));
#endif 

}

//-----------------------------------------------------------------------------
TEST(conduit_data_accessor, default_construct)
{
    index_t_accessor n_acc;
    Node n;
    n.set({-1,2,-3,4,-5});

    n_acc = n.value();
    EXPECT_EQ(n_acc[0],(index_t)(-1));
    EXPECT_EQ(n_acc[1],(index_t)( 2));
    EXPECT_EQ(n_acc[2],(index_t)(-3));
    EXPECT_EQ(n_acc[3],(index_t)( 4));
    EXPECT_EQ(n_acc[4],(index_t)(-5));
}






