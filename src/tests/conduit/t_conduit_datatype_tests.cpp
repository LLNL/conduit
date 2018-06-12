//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: type_tests.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"


using namespace conduit;


//-----------------------------------------------------------------------------
void print_dt(const DataType &dtype)
{
    std::cout << dtype.to_json() << std::endl;
}

//-----------------------------------------------------------------------------
TEST(dtype_tests, value_print)
{
    EXPECT_EQ(DataType::EMPTY_ID,0);
    EXPECT_EQ(DataType::id_to_name(DataType::EMPTY_ID),"empty");
    EXPECT_EQ(DataType::name_to_id("empty"),DataType::EMPTY_ID);
    EXPECT_TRUE( (DataType::EMPTY_ID != DataType::OBJECT_ID) );

    DataType dt;
    
    // empty leaf
    dt = DataType::empty();
    EXPECT_TRUE(dt.is_empty());
    print_dt(dt);

    // generic type
    dt = DataType::object();
    EXPECT_TRUE(dt.is_object());
    print_dt(dt);

    dt = DataType::list();
    EXPECT_TRUE(dt.is_list());
    print_dt(dt);

    // signed ints
    dt = DataType::int8();
    EXPECT_TRUE(dt.is_int8());
    print_dt(dt);

    dt = DataType::int16();
    EXPECT_TRUE(dt.is_int16());
    print_dt(dt);
    
    dt = DataType::int32();
    EXPECT_TRUE(dt.is_int32());
    print_dt(dt);

    dt = DataType::int64();
    EXPECT_TRUE(dt.is_int64());
    print_dt(dt);

    // unsigned ints
    dt = DataType::uint8();
    EXPECT_TRUE(dt.is_uint8());
    print_dt(dt);

    dt = DataType::uint16();
    EXPECT_TRUE(dt.is_uint16());
    print_dt(dt);
    
    dt = DataType::uint32();
    EXPECT_TRUE(dt.is_uint32());
    print_dt(dt);

    dt = DataType::uint64();
    EXPECT_TRUE(dt.is_uint64());
    print_dt(dt);
    
    // floating point
    dt = DataType::float32();
    EXPECT_TRUE(dt.is_float32());
    print_dt(dt);

    dt = DataType::float64();
    EXPECT_TRUE(dt.is_float64());
    print_dt(dt);

}

//-----------------------------------------------------------------------------
TEST(dtype_tests, c_types_value_print)
{
    
    DataType dt;
    
    // signed ints
    dt = DataType::c_char();
    EXPECT_TRUE(dt.is_char());
    print_dt(dt);
    
    dt = DataType::c_short();
    EXPECT_TRUE(dt.is_short());
    print_dt(dt);
    
    dt = DataType::c_int();
    EXPECT_TRUE(dt.is_int());
    print_dt(dt);

    dt = DataType::c_long();
    EXPECT_TRUE(dt.is_long());
    print_dt(dt);

#ifdef CONDUIT_HAS_LONG_LONG
    dt = DataType::c_long_long();
    EXPECT_TRUE(dt.is_long_long());
    print_dt(dt);
#else
    EXPECT_FALSE(dt.is_long_long());
#endif

    // unsigned ints
    dt = DataType::c_unsigned_char();
    EXPECT_TRUE(dt.is_unsigned_char());
    print_dt(dt);
    
    dt = DataType::c_unsigned_short();
    EXPECT_TRUE(dt.is_unsigned_short());
    print_dt(dt);
    
    dt = DataType::c_unsigned_int();
    EXPECT_TRUE(dt.is_unsigned_int());
    print_dt(dt);

    dt = DataType::c_unsigned_long();
    EXPECT_TRUE(dt.is_unsigned_long());
    print_dt(dt);

#ifdef CONDUIT_HAS_LONG_LONG
    dt = DataType::c_unsigned_long_long();
    EXPECT_TRUE(dt.is_unsigned_long_long());
    print_dt(dt);
#else
    EXPECT_FALSE(dt.is_unsigned_long_long());
#endif

    // floats
    dt = DataType::c_float();
    EXPECT_TRUE(dt.is_float());
    print_dt(dt);

    dt = DataType::c_double();
    EXPECT_TRUE(dt.is_double());
    print_dt(dt);
    
#ifdef CONDUIT_USE_LONG_DOUBLE
        dt = DataType::c_long_double();
        EXPECT_TRUE(dt.is_long_double());
        print_dt(dt);
#else
        // if we aren't using long double, this will always return false
        EXPECT_FALSE(dt.is_long_double());
#endif
}


//-----------------------------------------------------------------------------
TEST(dtype_tests, dtype_construct_from_string)
{
    print_dt(DataType("int8",
                      1,
                      0,
                      sizeof(int8),
                      sizeof(int8),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("int16",
                      1,
                      0,
                      sizeof(int16),
                      sizeof(int16),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("int32",
                      1,
                      0,
                      sizeof(int32),
                      sizeof(int32),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("int64",
                      1,
                      0,
                      sizeof(int64),
                      sizeof(int64),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("uint8",
                      1,
                      0,
                      sizeof(uint8),
                      sizeof(uint8),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("uint16",
                      1,
                      0,
                      sizeof(uint16),
                      sizeof(uint16),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("uint32",
                      1,
                      0,
                      sizeof(uint32),
                      sizeof(uint32),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("uint64",
                      1,
                      0,
                      sizeof(uint64),
                      sizeof(uint64),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("float32",
                      1,
                      0,
                      sizeof(float32),
                      sizeof(float32),
                      Endianness::DEFAULT_ID));

    print_dt(DataType("float64",
                      1,
                      0,
                      sizeof(float64),
                      sizeof(float64),
                      Endianness::DEFAULT_ID));

}

//-----------------------------------------------------------------------------
TEST(dtype_tests, dtype_set_using_string)
{
    DataType dt;
    
    EXPECT_EQ(DataType::EMPTY_ID,dt.id());
    
    dt.set("int8",
           1,
           0,
           sizeof(int8),
           sizeof(int8),
           Endianness::DEFAULT_ID);
    print_dt(dt);
    EXPECT_EQ(DataType::INT8_ID,dt.id());
    
    dt.set("int16",
           1,
           0,
           sizeof(int16),
           sizeof(int16),
           Endianness::DEFAULT_ID);
    print_dt(dt);
    EXPECT_EQ(DataType::INT16_ID,dt.id());
    
    dt.set("int32",
           1,
           0,
           sizeof(int32),
           sizeof(int32),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::INT32_ID,dt.id());
        
    dt.set("int64",
           1,
           0,
           sizeof(int64),
           sizeof(int64),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::INT64_ID,dt.id());
    
    
    dt.set("uint8",
           1,
           0,
           sizeof(uint8),
           sizeof(uint8),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::UINT8_ID,dt.id());
    
    dt.set("uint16",
           1,
           0,
           sizeof(uint16),
           sizeof(uint16),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::UINT16_ID,dt.id());
    
    dt.set("uint32",
           1,
           0,
           sizeof(uint32),
           sizeof(uint32),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::UINT32_ID,dt.id());
        
    dt.set("uint64",
           1,
           0,
           sizeof(uint64),
           sizeof(uint64),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::UINT64_ID,dt.id());
    
    dt.set("float32",
           1,
           0,
           sizeof(float32),
           sizeof(float32),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::FLOAT32_ID,dt.id());
        
    dt.set("float64",
           1,
           0,
           sizeof(float64),
           sizeof(float64),
           Endianness::DEFAULT_ID);
    
    print_dt(dt);
    EXPECT_EQ(DataType::FLOAT64_ID,dt.id());
    
    dt.set("char8_str",
           1,
           0,
           1,
           1,
           Endianness::DEFAULT_ID);

    print_dt(dt);
    EXPECT_EQ(DataType::CHAR8_STR_ID,dt.id());
    EXPECT_TRUE(dt.is_string());
    EXPECT_TRUE(dt.is_char8_str());
    
}


//-----------------------------------------------------------------------------
TEST(dtype_tests, default_bytes_from_string)
{

    EXPECT_EQ(0,DataType::default_bytes("[empty]"));
    EXPECT_EQ(0,DataType::default_bytes("object"));
    EXPECT_EQ(0,DataType::default_bytes("list"));

    
    EXPECT_EQ(sizeof(int8),DataType::default_bytes("int8"));
    EXPECT_EQ(sizeof(int16),DataType::default_bytes("int16"));
    EXPECT_EQ(sizeof(int32),DataType::default_bytes("int32"));
    EXPECT_EQ(sizeof(int64),DataType::default_bytes("int64"));
    
    
    EXPECT_EQ(sizeof(uint8),DataType::default_bytes("uint8"));
    EXPECT_EQ(sizeof(uint16),DataType::default_bytes("uint16"));
    EXPECT_EQ(sizeof(uint32),DataType::default_bytes("uint32"));
    EXPECT_EQ(sizeof(uint64),DataType::default_bytes("uint64"));

    EXPECT_EQ(sizeof(float32),DataType::default_bytes("float32"));
    EXPECT_EQ(sizeof(float64),DataType::default_bytes("float64"));

}



//-----------------------------------------------------------------------------
TEST(dtype_tests, default_dtype_from_string)
{

    EXPECT_TRUE(DataType::default_dtype("[empty]").is_empty());
    EXPECT_TRUE(DataType::default_dtype("object").is_object());
    EXPECT_TRUE(DataType::default_dtype("list").is_list());

    EXPECT_TRUE(DataType::default_dtype("int8").is_int8());
    EXPECT_TRUE(DataType::default_dtype("int16").is_int16());
    EXPECT_TRUE(DataType::default_dtype("int32").is_int32());
    EXPECT_TRUE(DataType::default_dtype("int64").is_int64());

    EXPECT_TRUE(DataType::default_dtype("uint8").is_uint8());
    EXPECT_TRUE(DataType::default_dtype("uint16").is_uint16());
    EXPECT_TRUE(DataType::default_dtype("uint32").is_uint32());
    EXPECT_TRUE(DataType::default_dtype("uint64").is_uint64());
    
    EXPECT_TRUE(DataType::default_dtype("float32").is_float32());
    EXPECT_TRUE(DataType::default_dtype("float64").is_float64());

}


//-----------------------------------------------------------------------------
TEST(dtype_tests,dtype_id_from_c_type_names)
{

    EXPECT_EQ(CONDUIT_NATIVE_CHAR_ID,  DataType::c_type_name_to_id("char"));
    EXPECT_EQ(CONDUIT_NATIVE_SHORT_ID, DataType::c_type_name_to_id("short"));
    EXPECT_EQ(CONDUIT_NATIVE_INT_ID,   DataType::c_type_name_to_id("int"));
    EXPECT_EQ(CONDUIT_NATIVE_LONG_ID,  DataType::c_type_name_to_id("long"));

#if CONDUIT_HAS_LONG_LONG
    EXPECT_EQ(CONDUIT_NATIVE_LONG_LONG_ID,
              DataType::c_type_name_to_id("long long"));
#endif

    EXPECT_EQ(CONDUIT_NATIVE_UNSIGNED_CHAR_ID,
              DataType::c_type_name_to_id("unsigned char"));

    EXPECT_EQ(CONDUIT_NATIVE_UNSIGNED_SHORT_ID,
              DataType::c_type_name_to_id("unsigned short"));

    EXPECT_EQ(CONDUIT_NATIVE_UNSIGNED_INT_ID,
              DataType::c_type_name_to_id("unsigned int"));

    EXPECT_EQ(CONDUIT_NATIVE_UNSIGNED_LONG_ID,
              DataType::c_type_name_to_id("unsigned long"));

#if CONDUIT_HAS_LONG_LONG
    EXPECT_EQ(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID,
              DataType::c_type_name_to_id("unsigned long long"));
#endif

    EXPECT_EQ(CONDUIT_NATIVE_FLOAT_ID, DataType::c_type_name_to_id("float"));
    EXPECT_EQ(CONDUIT_NATIVE_DOUBLE_ID,DataType::c_type_name_to_id("double"));

    EXPECT_EQ(CONDUIT_CHAR8_STR_ID,DataType::c_type_name_to_id("char8_str"));

}


//-----------------------------------------------------------------------------
TEST(dtype_tests,dtype_endianness_checks)
{

    DataType dt;
    dt.set(DataType::UINT64_ID,
           1,
           0,
           sizeof(uint64),
           sizeof(uint64),
           Endianness::DEFAULT_ID);

    dt.set(DataType::UINT64_ID,
           1,
           0,
           sizeof(uint64),
           sizeof(uint64),
           Endianness::BIG_ID);

    EXPECT_TRUE(dt.is_big_endian());

    dt.set(DataType::UINT64_ID,
           1,
           0,
           sizeof(uint64),
           sizeof(uint64),
           Endianness::LITTLE_ID);

    EXPECT_TRUE(dt.is_little_endian());
    
    dt.set_endianness(Endianness::machine_default());
    EXPECT_TRUE(dt.endianness_matches_machine());
    
    if(Endianness::machine_is_little_endian())
    {
        dt.set_endianness(Endianness::BIG_ID);
    }
    else
    {
        dt.set_endianness(Endianness::LITTLE_ID);
    }
    
    EXPECT_FALSE(dt.endianness_matches_machine());
}





