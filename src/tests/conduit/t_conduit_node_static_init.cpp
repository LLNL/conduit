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

//
// This test is here to trigger 
//  https://isocpp.org/wiki/faq/ctors#static-init-order
//  *If* it is present related to a conduit Node.
//
// Even with this test the crash may not always happen,
// but with several CI checks we increase our confidence
// that we are clear of static init problems
//


std::string use_with_static_init()
{
    conduit::Node n;
    n["value"] = 42;
    return n.to_yaml();
}

const std::string STATIC_INIT_RES = use_with_static_init();

//-----------------------------------------------------------------------------
TEST(conduit_smoke, test_static_init)
{
    conduit::Node n;
    n["value"] = 42;
    EXPECT_EQ(n.to_yaml(),STATIC_INIT_RES);
}
