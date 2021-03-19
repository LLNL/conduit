// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_transform.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
#include "conduit_relay_mpi.hpp"

#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

/// Transform Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_transform, generate_points)
{
    int par_rank = relay::mpi::rank(MPI_COMM_WORLD);
    int par_size = relay::mpi::size(MPI_COMM_WORLD);

    Node full_mesh;
    conduit::blueprint::mesh::examples::misc("adjsets", 3, 3, 0, full_mesh);

    const std::string src_cset_name = full_mesh.child(0).fetch("coordsets").child_names()[0];
    const std::string src_topo_name = full_mesh.child(0).fetch("topologies").child_names()[0];
    const std::string src_aset_name = full_mesh.child(0).fetch("adjsets").child_names()[0];
    const std::string dst_topo_name = "points";
    const std::string dst_aset_name = "point_adj";

    Node mesh, info;
    if(par_size == 1)
    {
        mesh.set_external(full_mesh);
    }
    else
    {
        std::ostringstream oss;
        oss << "domain" << par_rank;
        const std::string domain_name = oss.str();
        mesh.set_external(full_mesh[domain_name]);
    }
    EXPECT_TRUE(conduit::blueprint::mpi::verify("mesh", mesh, info, MPI_COMM_WORLD));

    Node s2dmap, d2smap;
    conduit::blueprint::mpi::mesh::generate_points(mesh,
                                                   src_aset_name,
                                                   dst_aset_name,
                                                   dst_topo_name,
                                                   s2dmap,
                                                   d2smap);
    EXPECT_TRUE(conduit::blueprint::mpi::verify("mesh", mesh, info, MPI_COMM_WORLD));

    const std::vector<const Node *> domains = conduit::blueprint::mesh::domains(mesh);

    { // Sanity Tests //
        for(const Node *domain_ptr : domains)
        {
            const Node &domain = *domain_ptr;

            // Verify Paths //
            EXPECT_TRUE(domain["adjsets"].has_path(dst_aset_name));
            EXPECT_TRUE(domain["topologies"].has_path(dst_topo_name));

            // Verify Path Pointers //
            EXPECT_EQ(domain["adjsets"][dst_aset_name]["topology"].as_string(), dst_topo_name);
            EXPECT_EQ(domain["topologies"][dst_topo_name]["coordset"].as_string(), src_cset_name);
        }
    }

    { // Adjacency Tests //
        for(const Node *domain_ptr : domains)
        {
            const Node &domain = *domain_ptr;

            const Node &src_aset_groups = domain["adjsets"][src_aset_name]["groups"];
            const Node &dst_aset_groups = domain["adjsets"][dst_aset_name]["groups"];
            EXPECT_FALSE(dst_aset_groups.diff(src_aset_groups, info));
        }
    }
}

/// Test Driver ///

int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
