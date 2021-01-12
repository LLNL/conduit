// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_matset_xforms.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_log.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_util_mesh, mesh_util_to_silo_basic)
{
    Node mesh;
    {
        blueprint::mesh::examples::basic("quads", 2, 2, 2, mesh);

        float64 mset_a_vfs[] = {1.0, 0.5, 0.5, 0.0};
        float64 mset_b_vfs[] = {0.0, 0.5, 0.5, 1.0};

        Node &mset = mesh["matsets/matset"];
        mset["topology"].set(mesh["topologies"].child_names().front());
        mset["volume_fractions/a"].set(&mset_a_vfs[0], 4);
        mset["volume_fractions/b"].set(&mset_b_vfs[0], 4);
    }
    Node &mset = mesh["matsets/matset"];

    Node silo, info;
    blueprint::mesh::matset::to_silo(mset, silo);
    std::cout << silo.to_yaml() << std::endl;

    { // Check General Contents //
        EXPECT_TRUE(silo.has_child("topology"));
        EXPECT_TRUE(silo.has_child("matlist"));
        EXPECT_TRUE(silo.has_child("mix_next"));
        EXPECT_TRUE(silo.has_child("mix_mat"));
        EXPECT_TRUE(silo.has_child("mix_vf"));
    }

    { // Check 'topology' Field //
        const std::string expected_topology = mset["topology"].as_string();
        const std::string actual_topology = silo["topology"].as_string();
        EXPECT_EQ(actual_topology, expected_topology);
    }

    // { // Check 'matlist' Field //
    //     // TODO(JRC): Need to make sure these are the same type.
    //     int64 expected_matlist_vec[] = {1, -1, -3, 2};
    //     Node expected_matlist(DataType::int64(4),
    //         &expected_matlist_vec[0], true);
    //     const Node &actual_matlist = silo["matlist"];

    //     EXPECT_FALSE(actual_matlist.diff(expected_matlist, info));
    // }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_util_mesh, mesh_util_venn_to_silo)
{
    const int nx = 4, ny = 4;
    const double radius = 0.25;

    CONDUIT_INFO("FULL VFS");
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("full", nx, ny, radius, mesh);
        const Node &mset = mesh["matsets/matset"];

        std::cout << mset.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::matset::to_silo(mset, mset_silo);
        std::cout << mset_silo.to_yaml() << std::endl;
    }

    CONDUIT_INFO("SPARSE BY MAT VFS");
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_material", nx, ny, radius, mesh);
        const Node &mset = mesh["matsets/matset"];

        std::cout << mset.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::matset::to_silo(mset, mset_silo);
        std::cout << mset_silo.to_yaml() << std::endl;
    }

    CONDUIT_INFO("SPARSE BY ELE VFS");
    {
        Node mesh, info;
        blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, mesh);
        const Node &mset = mesh["matsets/matset"];

        std::cout << mset.to_yaml() << std::endl;

        Node mset_silo;
        blueprint::mesh::matset::to_silo(mset, mset_silo);
        std::cout << mset_silo.to_yaml() << std::endl;
    }

}
