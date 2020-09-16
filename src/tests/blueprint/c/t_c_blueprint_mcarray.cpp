// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_c_blueprint_mcarray.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"
#include "conduit_blueprint.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_blueprint_mcarray, create_and_verify)
{
    conduit_node *n      = conduit_node_create();
    conduit_node *nxform = conduit_node_create();
    conduit_node *nempty = conduit_node_create();
    conduit_node *info   = conduit_node_create();

    conduit_blueprint_mcarray_examples_xyz("interleaved",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));
    EXPECT_TRUE(conduit_blueprint_mcarray_is_interleaved(n));

    EXPECT_TRUE(conduit_blueprint_mcarray_to_contiguous(n,nxform));
    EXPECT_FALSE(conduit_blueprint_mcarray_is_interleaved(nxform));
    EXPECT_TRUE(conduit_node_is_contiguous(nxform));

    conduit_blueprint_mcarray_examples_xyz("separate",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));

    conduit_blueprint_mcarray_examples_xyz("contiguous",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));
    EXPECT_TRUE(conduit_node_is_contiguous(n));
    EXPECT_FALSE(conduit_blueprint_mcarray_is_interleaved(n));

    EXPECT_TRUE(conduit_blueprint_mcarray_to_interleaved(n,nxform));
    conduit_node_print_detailed(nxform);
    EXPECT_TRUE(conduit_blueprint_mcarray_is_interleaved(nxform));

    conduit_blueprint_mcarray_examples_xyz("interleaved_mixed",10,n);
    EXPECT_TRUE(conduit_blueprint_mcarray_verify(n,info));
    
    EXPECT_FALSE(conduit_blueprint_mcarray_verify_sub_protocol("sub",n,info));
    EXPECT_FALSE(conduit_blueprint_mcarray_verify(nempty,info));

    conduit_node_destroy(n);
    conduit_node_destroy(nxform);
    conduit_node_destroy(nempty);
    conduit_node_destroy(info);
}


