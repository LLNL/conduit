// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_query.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
#include "conduit_relay_mpi.hpp"

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

/// Query Tests ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mpi_mesh_transform, find_delegate_domain)
{
    const int par_rank = relay::mpi::rank(MPI_COMM_WORLD);
    // const int par_size = relay::mpi::size(MPI_COMM_WORLD);

    Node info;

    { // Empty Node Test //
        Node mesh, delegate;
        blueprint::mpi::mesh::find_delegate_domain(mesh, delegate, MPI_COMM_WORLD);

        ASSERT_TRUE(blueprint::mesh::verify(delegate, info));
        ASSERT_TRUE(delegate.dtype().is_empty());
    }

    { // No Empty Node Test //
        Node mesh, delegate;
        conduit::blueprint::mesh::examples::basic("quads", 2, 2, 0, mesh);
        blueprint::mpi::mesh::find_delegate_domain(mesh, delegate, MPI_COMM_WORLD);

        ASSERT_TRUE(blueprint::mesh::verify(delegate, info));
        ASSERT_FALSE(delegate.dtype().is_empty());
        ASSERT_FALSE(delegate.diff(mesh, info));
    }

    { // Some Empty Node Test //
        Node full_mesh, rank_mesh, delegate;
        conduit::blueprint::mesh::examples::basic("quads", 2, 2, 0, full_mesh);
        if(par_rank == 0)
        {
            rank_mesh.set_external(full_mesh);
        }

        blueprint::mpi::mesh::find_delegate_domain(rank_mesh, delegate, MPI_COMM_WORLD);

        ASSERT_TRUE(blueprint::mesh::verify(delegate, info));
        ASSERT_FALSE(delegate.dtype().is_empty());
        ASSERT_FALSE(delegate.diff(full_mesh, info));
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
