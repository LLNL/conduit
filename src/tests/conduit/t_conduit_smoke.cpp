// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(conduit_smoke, basic_use)
{
    EXPECT_EQ(sizeof(conduit::uint32),4);
    EXPECT_EQ(sizeof(conduit::uint64),8);
    EXPECT_EQ(sizeof(conduit::float64),8);
    
    std::cout << conduit::about() << std::endl;
    
    conduit::Node n1;
    n1["path"] = "mydata";
    conduit::Node n2;
    n2.set_external(n1);
    n1["path"].set(n2["path"]);
    // bad!
    n1.print();

}
