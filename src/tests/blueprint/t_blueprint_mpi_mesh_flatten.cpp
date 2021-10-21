// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_mpi_flatten.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include <string>

#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_blueprint_mpi.hpp>
#include <conduit_relay.hpp>
#include <conduit_relay_mpi.hpp>

#include <mpi.h>
#include "gtest/gtest.h"

// Enable this macro to generate baselines.
//#define GENERATE_BASELINES

//-----------------------------------------------------------------------------
#ifdef _WIN32
const std::string sep("\\");
#else
const std::string sep("/");
#endif

//-----------------------------------------------------------------------------
std::string
baseline_dir()
{
    std::string path(__FILE__);
    auto idx = path.rfind(sep);
    if(idx != std::string::npos)
        path = path.substr(0, idx);
    path = path + sep + std::string("baselines");
    return path;
}

//-----------------------------------------------------------------------------
std::string test_name() { return std::string("t_blueprint_mpi_mesh_flatten"); }

//-----------------------------------------------------------------------------
int
get_rank()
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

//-----------------------------------------------------------------------------
void
barrier()
{
    MPI_Barrier(MPI_COMM_WORLD);
}

//-----------------------------------------------------------------------------
// Include some helper function definitions
#include "t_blueprint_partition_helpers.hpp"

using namespace conduit;

TEST(t_blueprint_mpi_mesh_flatten, braid)
{
    const MPI_Comm comm = MPI_COMM_WORLD;
    Node mesh;
    blueprint::mpi::mesh::examples::braid_uniform_multi_domain(mesh, comm);

    Node table, opts;
    blueprint::mpi::mesh::flatten(mesh, opts, table, comm);

    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    if(rank == 0)
    {
        std::cout << table.to_json() << std::endl;
    }
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size == 4)
    {
        result = RUN_ALL_TESTS();
    }
    else
    {
        std::cout << "This program requires 4 ranks." << std::endl;
        result = -1;
    }
    MPI_Finalize();

    return result;
}
