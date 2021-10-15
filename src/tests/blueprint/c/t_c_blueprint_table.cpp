// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_c_blueprint_table.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"
#include "conduit_blueprint.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_blueprint_table, basic_and_verify)
{
    conduit_node *n      = conduit_node_create();
    conduit_node *info   = conduit_node_create();

    // Empty node should fail
    EXPECT_FALSE(conduit_blueprint_table_verify(n, info));
    EXPECT_FALSE(conduit_blueprint_verify("table", n, info));

    // Basic example should pass
    conduit_blueprint_table_examples_basic(5, 4, 3, n);
    EXPECT_TRUE(conduit_blueprint_table_verify(n, info));
    EXPECT_TRUE(conduit_blueprint_verify("table", n, info));

    // Subprotocol "subproto" should fail
    EXPECT_FALSE(conduit_blueprint_table_verify_sub_protocol("subproto", n, info));
    EXPECT_FALSE(conduit_blueprint_verify("table/subproto", n, info));

    conduit_node_destroy(n);
    conduit_node_destroy(info);
}
