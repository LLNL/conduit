// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_node_parent.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_parent, simple)
{
    uint32   val1  = 10;
    uint32   val2  = 20;

    Node n;
    n["a"]["b"] = val1;
    n["a"]["c"] = val2;
    EXPECT_EQ(n["a"]["b"].as_uint32(),val1);
    EXPECT_EQ(n["a"]["c"].as_uint32(),val2);
    Node &b = n["a/c"];
    EXPECT_EQ(b["../b"].as_uint32(),val1);
    EXPECT_EQ(n["a/c/../b"].as_uint32(),val1);
    
}


