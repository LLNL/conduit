// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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

