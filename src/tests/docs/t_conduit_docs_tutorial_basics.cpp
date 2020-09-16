// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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

