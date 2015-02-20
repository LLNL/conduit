//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
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

