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
/// file: t_conduit_docs_tutorial_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "t_conduit_docs_tutorial_helpers.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_very_basic)
{
    BEGIN_EXAMPLE("basics_very_basic");
    Node n;
    n["my"] = "data";
    n.print(); 
    END_EXAMPLE("basics_very_basic");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_hierarchial)
{
    BEGIN_EXAMPLE("basics_hierarchial");
    Node n;
    n["my"] = "data";
    n["a/b/c"] = "d";
    n["a"]["b"]["e"] = 64.0;
    n.print();

    std::cout << "total bytes: " << n.total_strided_bytes() << std::endl;
    END_EXAMPLE("basics_hierarchial");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_object_and_list)
{
    BEGIN_EXAMPLE("basics_object_and_list");
    Node n;
    n["object_example/val1"] = "data";
    n["object_example/val2"] = 10u;
    n["object_example/val3"] = 3.1415;
    
    for(int i = 0; i < 5 ; i++ )
    {
        Node &list_entry = n["list_example"].append();
        list_entry.set(i);
    }
    
    n.print();
    END_EXAMPLE("basics_object_and_list");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_object_and_list_itr)
{
    BEGIN_EXAMPLE("basics_object_and_list_itr");
    Node n;
    n["object_example/val1"] = "data";
    n["object_example/val2"] = 10u;
    n["object_example/val3"] = 3.1415;
    
    for(int i = 0; i < 5 ; i++ )
    {
        Node &list_entry = n["list_example"].append();
        list_entry.set(i);
    }

    n.print();

    NodeIterator itr = n["object_example"].children();
    while(itr.has_next())
    {
        Node &cld = itr.next();
        std::string cld_name = itr.name();
        std::cout << cld_name << ": " << cld.to_string() << std::endl;
    }

    std::cout << std::endl;

    itr = n["list_example"].children();
    while(itr.has_next())
    {
        Node &cld = itr.next();
        std::cout << cld.to_string() << std::endl;
    }
    END_EXAMPLE("basics_object_and_list_itr");
}


//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_mem_spaces)
{
    BEGIN_EXAMPLE("basics_mem_spaces");
    Node n;
    n["my"] = "data";
    n["a/b/c"] = "d";
    n["a"]["b"]["e"] = 64.0;

    Node ninfo;
    n.info(ninfo);
    ninfo.print();
    END_EXAMPLE("basics_mem_spaces");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_bw_style)
{
    BEGIN_EXAMPLE("basics_bw_style");
    Node n;
    uint32 val = 100;
    n["test"] = val;
    n.print();
    n.print_detailed();
    END_EXAMPLE("basics_bw_style");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, basics_bw_style_from_native)
{
    BEGIN_EXAMPLE("basics_bw_style_from_native");
    Node n;
    int val = 100;
    n["test"] = val;
    n.print_detailed();
    END_EXAMPLE("basics_bw_style_from_native");
}

