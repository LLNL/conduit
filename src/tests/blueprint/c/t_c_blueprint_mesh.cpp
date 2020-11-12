// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_c_blueprint_mesh.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.h"
#include "conduit_blueprint.h"

#include <stdio.h>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(c_conduit_blueprint_mesh, create_and_verify)
{
    conduit_node *n      = conduit_node_create();
    conduit_node *nindex = conduit_node_create();
    conduit_node *nempty = conduit_node_create();
    conduit_node *info   = conduit_node_create();

    EXPECT_FALSE(conduit_blueprint_mesh_verify(nempty,info));
    

    conduit_blueprint_mesh_examples_braid("hexs",3,3,3,n);
    EXPECT_TRUE(conduit_blueprint_mesh_verify(n,info));
    
    conduit_node *ntopo = conduit_node_fetch(n,"topologies/mesh");
    EXPECT_TRUE(conduit_blueprint_mesh_verify_sub_protocol("topology",ntopo,info));
    EXPECT_FALSE(conduit_blueprint_mesh_verify_sub_protocol("coordset",ntopo,info));
        
    conduit_blueprint_mesh_generate_index(n,"",1,nindex);

    EXPECT_TRUE(conduit_blueprint_mesh_verify_sub_protocol("index",nindex,info));

    conduit_node_destroy(n);
    conduit_node_destroy(nindex);
    conduit_node_destroy(nempty);
    conduit_node_destroy(info);
}


