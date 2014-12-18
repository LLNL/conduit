//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// file: numpy_smoke.cpp
///


#include "Python.h"   
#include "numpy/npy_common.h"


#include <iostream>
#include "gtest/gtest.h"


TEST(numpy_smoke_test, numpy_smoke)
{
    EXPECT_EQ(sizeof(npy_uint32),4);
    EXPECT_EQ(sizeof(npy_uint64),8);
    EXPECT_EQ(sizeof(npy_float64),8);

    // test 32 bit unsigned mask
    npy_uint32 uival_32 = 0xFFFFFFFF;
    npy_uint64 uival_64 = 0x100000000;
    
    EXPECT_EQ(uival_32,4294967295);
    EXPECT_EQ(uival_64,4294967296);

}
