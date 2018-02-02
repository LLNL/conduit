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
/// file: conduit_char8_str.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_char8_str, basic)
{
    const char *c_ta = "test string for a";
    const char *c_tb = "test string for b";

    Node n;
    n["a"] = c_ta;
    n["b"] = c_tb;
    
    std::ostringstream oss;
    oss << n["a"].as_string() << " and " << n["b"].as_string();
    
    std::string cpp_tc = oss.str();
    
    n["c"] = cpp_tc;

    EXPECT_EQ(strcmp(n["a"].as_char8_str(),c_ta),0);
    EXPECT_EQ(strcmp(n["b"].as_char8_str(),c_tb),0);
    EXPECT_EQ(n["c"].as_string(),cpp_tc);

    n.print_detailed();
}


//-----------------------------------------------------------------------------
TEST(conduit_char8_str, list_of_per_alloc)
{
    Node n;
    n.list_of(DataType::char8_str(10),5);
 
    n[0].set("a");
    n[1].set("bb");
    n[2].set("ccc");
    n[3].set("bbbb");
    n[4].set("ddddd");
    
    n.print();

    EXPECT_EQ(n.number_of_children(),5);
 
    Node n_info;
    n.info(n_info);
    n_info.print();
    
    NodeConstIterator itr = n_info["mem_spaces"].children();

    int alloc_count = 0;
    while(itr.has_next())
    {
        const Node &curr = itr.next();
        if( curr["type"].as_string() == std::string("allocated") )
        {
            alloc_count++;
        }
    }
    
    // make sure there is only one alloc
    EXPECT_EQ(alloc_count,1);
}

