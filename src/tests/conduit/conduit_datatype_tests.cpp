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
TEST(type_tests, value_print)
{
    EXPECT_EQ(DataType::EMPTY_T,0);
    EXPECT_EQ(DataType::id_to_name(DataType::EMPTY_T),"[empty]");
    EXPECT_EQ(DataType::name_to_id("[empty]"),DataType::EMPTY_T);
    EXPECT_TRUE( (DataType::EMPTY_T != DataType::OBJECT_T) );

    print_dt(DataType::empty());
    print_dt(DataType::object());
    print_dt(DataType::list());
    
    print_dt(DataType::int8());
    print_dt(DataType::int16());
    print_dt(DataType::int32());
    print_dt(DataType::int64());

    print_dt(DataType::uint8());
    print_dt(DataType::uint16());
    print_dt(DataType::uint32());
    print_dt(DataType::uint64());

    print_dt(DataType::float32());
    print_dt(DataType::float64());

}

//-----------------------------------------------------------------------------
TEST(type_tests, c_types_value_print)
{
    
    print_dt(DataType::c_char());
    print_dt(DataType::c_short());
    print_dt(DataType::c_int());
    print_dt(DataType::c_long());

    print_dt(DataType::c_unsigned_char());
    print_dt(DataType::c_unsigned_short());
    print_dt(DataType::c_unsigned_int());
    print_dt(DataType::c_unsigned_long());

    print_dt(DataType::c_float());
    print_dt(DataType::c_double());
}


