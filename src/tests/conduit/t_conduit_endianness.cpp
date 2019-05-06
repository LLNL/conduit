//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_endianness.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;


//-----------------------------------------------------------------------------
TEST(conduit_endianness, endianness_ids_and_names)
{
    if(Endianness::machine_is_big_endian())
    {
        EXPECT_TRUE(Endianness::machine_default() == Endianness::BIG_ID);
    }
    
    if(Endianness::machine_is_little_endian())
    {
        EXPECT_TRUE(Endianness::machine_default() == Endianness::LITTLE_ID);
    }

    EXPECT_EQ(Endianness::name_to_id("big"),Endianness::BIG_ID );
    EXPECT_EQ(Endianness::name_to_id("little"),Endianness::LITTLE_ID );

    EXPECT_EQ(Endianness::id_to_name(Endianness::BIG_ID),"big");
    EXPECT_EQ(Endianness::id_to_name(Endianness::LITTLE_ID),"little");

}

//-----------------------------------------------------------------------------
TEST(conduit_endianness, simple_1)
{
    union{uint8  vbytes[4]; uint32 vuint;} test;
    std::string machine_endian = Endianness::id_to_name(Endianness::machine_default());
    std::cout << "[host is " << machine_endian << "]" << std::endl;

    if(Endianness::machine_default() == Endianness::BIG_ID)
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

//-----------------------------------------------------------------------------
TEST(conduit_endianness, swap_inplace)
{
    union{uint8  vbytes[2]; uint16 vuint16;} test16;
    union{uint8  vbytes[4]; uint32 vuint32;} test32;
    union{uint8  vbytes[8]; uint64 vuint64;} test64;
        
    if(Endianness::machine_default() == Endianness::BIG_ID)
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

//-----------------------------------------------------------------------------
TEST(conduit_endianness, swap_to)
{
    union{uint8  vbytes[2]; uint16 vuint16;} src_test16;
    union{uint8  vbytes[4]; uint32 vuint32;} src_test32;
    union{uint8  vbytes[8]; uint64 vuint64;} src_test64;
    
    uint16 dest_test16;
    uint32 dest_test32;
    uint64 dest_test64;
        
    if(Endianness::machine_default() == Endianness::BIG_ID)
    {
     
        src_test16.vbytes[0] =  0x02;
        src_test16.vbytes[1] =  0x01;

        Endianness::swap16(&src_test16.vuint16,
                           &dest_test16);
                           
        EXPECT_EQ(0x0102,dest_test16);

        src_test32.vbytes[0] =  0x04;
        src_test32.vbytes[1] =  0x03;
        src_test32.vbytes[2] =  0x02;
        src_test32.vbytes[3] =  0x01;

        Endianness::swap32(&src_test32.vuint32,
                           &dest_test32);

        EXPECT_EQ(0x01020304,dest_test32);

        src_test64.vbytes[0] =  0x08;
        src_test64.vbytes[1] =  0x07;
        src_test64.vbytes[2] =  0x06;
        src_test64.vbytes[3] =  0x05;
        src_test64.vbytes[4] =  0x04;
        src_test64.vbytes[5] =  0x03;
        src_test64.vbytes[6] =  0x02;
        src_test64.vbytes[7] =  0x01;

        Endianness::swap64(&src_test64.vuint64,
                           &dest_test64);
                           
        EXPECT_EQ(0x0102030405060708,dest_test64);
    }
    else
    {
        src_test16.vbytes[0] =  0x01;
        src_test16.vbytes[1] =  0x02;

        Endianness::swap16(&src_test16.vuint16,
                           &dest_test16);
        EXPECT_EQ(0x0102,dest_test16);

        src_test32.vbytes[0] =  0x01;
        src_test32.vbytes[1] =  0x02;
        src_test32.vbytes[2] =  0x03;
        src_test32.vbytes[3] =  0x04;

        Endianness::swap32(&src_test32.vuint32,
                           &dest_test32);

        EXPECT_EQ(0x01020304,dest_test32);

        src_test64.vbytes[0] =  0x01;
        src_test64.vbytes[1] =  0x02;
        src_test64.vbytes[2] =  0x03;
        src_test64.vbytes[3] =  0x04;
        src_test64.vbytes[4] =  0x05;
        src_test64.vbytes[5] =  0x06;
        src_test64.vbytes[6] =  0x07;
        src_test64.vbytes[7] =  0x08;

        Endianness::swap64(&src_test64.vuint64,
                           &dest_test64);
        EXPECT_EQ(0x0102030405060708,dest_test64);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_endianness, node_swap)
{
    union{uint8  vbytes[2]; uint16 vuint16;} test16;
    union{uint8  vbytes[4]; uint32 vuint32;} test32;
    union{uint8  vbytes[8]; uint64 vuint64;} test64;

    Node n;

    if(Endianness::machine_default() == Endianness::BIG_ID)
    {
        test16.vbytes[0] =  0x02;
        test16.vbytes[1] =  0x01;

        n["test16"].set_external(&test16.vuint16,1);
        /// no swap if types match (machine default will be BIG_ID)
        n["test16"].endian_swap(Endianness::DEFAULT_ID);
        EXPECT_EQ(0x0201,test16.vuint16);
        /// no swap if types match (machine default will be BIG_ID)
        n["test16"].endian_swap(Endianness::BIG_ID);
        EXPECT_EQ(0x0201,test16.vuint16);
        /// swap
        n["test16"].endian_swap(Endianness::LITTLE_ID);
        EXPECT_EQ(0x0102,test16.vuint16);

        test32.vbytes[0] =  0x04;
        test32.vbytes[1] =  0x03;
        test32.vbytes[2] =  0x02;
        test32.vbytes[3] =  0x01;

        n["test32"].set_external(&test32.vuint32,1);
        n["test32"].endian_swap(Endianness::LITTLE_ID);
        EXPECT_EQ(0x01020304,test32.vuint32);
        
        test64.vbytes[0] =  0x08;
        test64.vbytes[1] =  0x07;
        test64.vbytes[2] =  0x06;
        test64.vbytes[3] =  0x05;
        test64.vbytes[4] =  0x04;
        test64.vbytes[5] =  0x03;
        test64.vbytes[6] =  0x02;
        test64.vbytes[7] =  0x01;

        n["test64"].set_external(&test64.vuint64,1);
        n["test64"].endian_swap(Endianness::LITTLE_ID);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

        // swap all back back so we can do a full test on n
        n["test16"].endian_swap(Endianness::BIG_ID);
        n["test32"].endian_swap(Endianness::BIG_ID);
        n["test64"].endian_swap(Endianness::BIG_ID);
        
        // full swap via n
        n.endian_swap(Endianness::LITTLE_ID);
        
        EXPECT_EQ(0x0102,test16.vuint16);
        EXPECT_EQ(0x01020304,test32.vuint32);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

    }
    else
    {
        test16.vbytes[0] =  0x01;
        test16.vbytes[1] =  0x02;

        n["test16"].set_external(&test16.vuint16,1);
        /// no swap if types match (machine default will be LITTLE_ID)
        n["test16"].endian_swap(Endianness::DEFAULT_ID);
        EXPECT_EQ(0x0201,test16.vuint16);
        /// no swap if types match (machine default will be LITTLE_ID)
        n["test16"].endian_swap(Endianness::LITTLE_ID);
        EXPECT_EQ(0x0201,test16.vuint16);
        /// swap
        n["test16"].endian_swap(Endianness::BIG_ID);
        EXPECT_EQ(0x0102,test16.vuint16);

        test32.vbytes[0] =  0x01;
        test32.vbytes[1] =  0x02;
        test32.vbytes[2] =  0x03;
        test32.vbytes[3] =  0x04;

        n["test32"].set_external(&test32.vuint32,1);
        n["test32"].endian_swap(Endianness::BIG_ID);
        EXPECT_EQ(0x01020304,test32.vuint32);

        test64.vbytes[0] =  0x01;
        test64.vbytes[1] =  0x02;
        test64.vbytes[2] =  0x03;
        test64.vbytes[3] =  0x04;
        test64.vbytes[4] =  0x05;
        test64.vbytes[5] =  0x06;
        test64.vbytes[6] =  0x07;
        test64.vbytes[7] =  0x08;

        n["test64"].set_external(&test64.vuint64,1);
        n["test64"].endian_swap(Endianness::BIG_ID);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

        // swap all back back so we can do a full test on n
        n["test16"].endian_swap(Endianness::LITTLE_ID);
        n["test32"].endian_swap(Endianness::LITTLE_ID);
        n["test64"].endian_swap(Endianness::LITTLE_ID);
        n.endian_swap(Endianness::BIG_ID);
        
        // full swap via n
        EXPECT_EQ(0x0102,test16.vuint16);
        EXPECT_EQ(0x01020304,test32.vuint32);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

    }
}

//-----------------------------------------------------------------------------
TEST(conduit_endianness, node_swap_using_explicit_funcs)
{
    union{uint8  vbytes[2]; uint16 vuint16;} test16;
    union{uint8  vbytes[4]; uint32 vuint32;} test32;
    union{uint8  vbytes[8]; uint64 vuint64;} test64;

    Node n;

    if(Endianness::machine_default() == Endianness::BIG_ID)
    {
        test16.vbytes[0] =  0x02;
        test16.vbytes[1] =  0x01;

        n["test16"].set_external(&test16.vuint16,
                                 1, // num ele
                                 0, // offset
                                 2, // stride
                                 2, // elem_bytes
                                 Endianness::LITTLE_ID); // set not match
        
        n["test16"].endian_swap_to_machine_default();
        EXPECT_EQ(0x0102,test16.vuint16);

        test32.vbytes[0] =  0x04;
        test32.vbytes[1] =  0x03;
        test32.vbytes[2] =  0x02;
        test32.vbytes[3] =  0x01;

        n["test32"].set_external(&test32.vuint32,
                                 1, // num ele
                                 0, // offset
                                 4, // stride
                                 4, // elem_bytes
                                 Endianness::LITTLE_ID); // set not match

        n["test32"].endian_swap_to_machine_default();
        EXPECT_EQ(0x01020304,test32.vuint32);
        
        test64.vbytes[0] =  0x08;
        test64.vbytes[1] =  0x07;
        test64.vbytes[2] =  0x06;
        test64.vbytes[3] =  0x05;
        test64.vbytes[4] =  0x04;
        test64.vbytes[5] =  0x03;
        test64.vbytes[6] =  0x02;
        test64.vbytes[7] =  0x01;

        n["test64"].set_external(&test64.vuint64,
                                 1, // num ele
                                 0, // offset
                                 8, // stride
                                 8, // elem_bytes
                                 Endianness::LITTLE_ID); // set not match

        n["test64"].endian_swap_to_machine_default();
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

        // swap all back back so we can do a full test on n
        n["test16"].endian_swap_to_little();
        n["test32"].endian_swap_to_little();
        n["test64"].endian_swap_to_little();
        // full swap via n
        n.endian_swap_to_big();
        
        EXPECT_EQ(0x0102,test16.vuint16);
        EXPECT_EQ(0x01020304,test32.vuint32);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

    }
    else
    {
        test16.vbytes[0] =  0x01;
        test16.vbytes[1] =  0x02;

        n["test16"].set_external(&test16.vuint16,
                                 1, // num ele
                                 0, // offset
                                 2, // stride
                                 2, // elem_bytes
                                 Endianness::BIG_ID); // set not match

        /// swap
        n["test16"].endian_swap_to_machine_default();
        EXPECT_EQ(0x0102,test16.vuint16);

        test32.vbytes[0] =  0x01;
        test32.vbytes[1] =  0x02;
        test32.vbytes[2] =  0x03;
        test32.vbytes[3] =  0x04;

        n["test32"].set_external(&test32.vuint32,
                                 1, // num ele
                                 0, // offset
                                 4, // stride
                                 4, // elem_bytes
                                 Endianness::BIG_ID); // set not match

        n["test32"].endian_swap_to_machine_default();
        EXPECT_EQ(0x01020304,test32.vuint32);

        test64.vbytes[0] =  0x01;
        test64.vbytes[1] =  0x02;
        test64.vbytes[2] =  0x03;
        test64.vbytes[3] =  0x04;
        test64.vbytes[4] =  0x05;
        test64.vbytes[5] =  0x06;
        test64.vbytes[6] =  0x07;
        test64.vbytes[7] =  0x08;

        n["test64"].set_external(&test64.vuint64,
                                 1, // num ele
                                 0, // offset
                                 8, // stride
                                 8, // elem_bytes
                                 Endianness::BIG_ID); // set not match

                                 
        
        n["test64"].endian_swap_to_machine_default();
        EXPECT_EQ(0x0102030405060708,test64.vuint64);

        // swap all back back so we can do a full test on n
        n["test16"].endian_swap_to_big();
        n["test32"].endian_swap_to_big();
        n["test64"].endian_swap_to_big();

        // full swap via n
        n.endian_swap_to_little();
        
        EXPECT_EQ(0x0102,test16.vuint16);
        EXPECT_EQ(0x01020304,test32.vuint32);
        EXPECT_EQ(0x0102030405060708,test64.vuint64);


    }
}


