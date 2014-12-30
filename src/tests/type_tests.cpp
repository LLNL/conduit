//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: type_tests.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"


using namespace conduit;


//-----------------------------------------------------------------------------
void print_dt(const DataType &dtype)
{
    std::cout << dtype.to_json() << std::endl;
}

//-----------------------------------------------------------------------------
TEST(type_tests, value_print)
{
    EXPECT_EQ(DataType::EMPTY_T,0);
    EXPECT_EQ(DataType::id_to_name(DataType::EMPTY_T),"[empty]");
    EXPECT_EQ(DataType::name_to_id("[empty]"),DataType::EMPTY_T);
    EXPECT_TRUE( (DataType::EMPTY_T != DataType::OBJECT_T) );

    print_dt(DataType::Objects::empty());
    print_dt(DataType::Objects::object());
    print_dt(DataType::Objects::list());
    
    print_dt(DataType::Scalars::int8());
    print_dt(DataType::Scalars::int16());
    print_dt(DataType::Scalars::int32());
    print_dt(DataType::Scalars::int64());

    print_dt(DataType::Scalars::uint8());
    print_dt(DataType::Scalars::uint16());
    print_dt(DataType::Scalars::uint32());
    print_dt(DataType::Scalars::uint64());

    print_dt(DataType::Scalars::float32());
    print_dt(DataType::Scalars::float64());

}

