// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_table_examples.cpp
///
//-----------------------------------------------------------------------------

#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_blueprint_table_examples.hpp>

#include <gtest/gtest.h>

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_table_examples, basic)
{
    Node basic;
    blueprint::table::examples::basic(10, 9, 8, basic);

    Node info;
    bool res = blueprint::table::verify(basic, info);
    ASSERT_TRUE(res) << info.to_json();
    EXPECT_EQ(10*9*8, info["values/rows"].to_index_t());
}
