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
/// file: conduit_char8_str.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"

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
