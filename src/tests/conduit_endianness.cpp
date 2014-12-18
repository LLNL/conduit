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
/// file: conduit_endianness.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;


TEST(conduit_endianness_simple_1, conduit_endianness)
{
    union{uint8  vbytes[4]; uint32 vuint;} test;
    std::string machine_endian = Endianness::id_to_name(Endianness::machine_default());
    std::cout << "[host is " << machine_endian << "]" << std::endl;

    if(Endianness::machine_default() == Endianness::BIG_T)
    {
        test.vbytes[0] =  0xff;
        test.vbytes[1] =  0xff;
        test.vbytes[2] =  0xff;
        test.vbytes[3] =  0xfe;
        
        EXPECT_EQ(0xfffffffe,test.vuint);
    }
    else
    {
        test.vbytes[0] =  0xfe;
        test.vbytes[1] =  0xff;
        test.vbytes[2] =  0xff;
        test.vbytes[3] =  0xff;
        
        EXPECT_EQ(0xfffffffe,test.vuint);  
    }
}

TEST(conduit_endianness_swap_inplace, conduit_endianness)
{
    union{uint8  vbytes[2]; uint16 vuint16;} test16;
    union{uint8  vbytes[4]; uint32 vuint32;} test32;
    union{uint8  vbytes[8]; uint64 vuint64;} test64;
        
    if(Endianness::machine_default() == Endianness::BIG_T)
    {
     
        test16.vbytes[0] =  0x02;
        test16.vbytes[1] =  0x01;

        Endianness::swap16(&test16.vuint16);
        EXPECT_EQ(0x0102,test16.vuint16);

        test32.vbytes[0] =  0x04;
        test32.vbytes[1] =  0x03;
        test32.vbytes[2] =  0x02;
        test32.vbytes[3] =  0x01;        

        Endianness::swap32(&test32.vuint32);
        EXPECT_EQ(0x01020304,test32.vuint32);

        test64.vbytes[0] =  0x08;
        test64.vbytes[1] =  0x07;
        test64.vbytes[2] =  0x06;
        test64.vbytes[3] =  0x05;        
        test64.vbytes[4] =  0x04;
        test64.vbytes[5] =  0x03;
        test64.vbytes[6] =  0x02;
        test64.vbytes[7] =  0x01;        

        Endianness::swap64(&test64.vuint64);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);  
    }
    else
    {
        test16.vbytes[0] =  0x01;
        test16.vbytes[1] =  0x02;

        Endianness::swap16(&test16.vuint16);
        EXPECT_EQ(0x0102,test16.vuint16);

        test32.vbytes[0] =  0x01;
        test32.vbytes[1] =  0x02;
        test32.vbytes[2] =  0x03;
        test32.vbytes[3] =  0x04;        

        Endianness::swap32(&test32.vuint32);
        EXPECT_EQ(0x01020304,test32.vuint32);

        test64.vbytes[0] =  0x01;
        test64.vbytes[1] =  0x02;
        test64.vbytes[2] =  0x03;
        test64.vbytes[3] =  0x04;        
        test64.vbytes[4] =  0x05;
        test64.vbytes[5] =  0x06;
        test64.vbytes[6] =  0x07;
        test64.vbytes[7] =  0x08;        

        Endianness::swap64(&test64.vuint64);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

    }
}
