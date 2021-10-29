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
#include <conduit_relay_mpi_io_blueprint.hpp>

#include <mpi.h>
#include "gtest/gtest.h"

#include "blueprint_test_helpers.hpp"

// Enable this macro to generate baselines.
// #define GENERATE_BASELINES

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
#include "blueprint_baseline_helpers.hpp"

using namespace conduit;

TEST(t_blueprint_mpi_mesh_flatten, braid)
{
    const MPI_Comm comm = MPI_COMM_WORLD;
    Node mesh;
    blueprint::mpi::mesh::examples::braid_uniform_multi_domain(mesh, comm);

#ifdef WRITE_MESH
    relay::mpi::io::blueprint::save_mesh(mesh, "flatten_braid.yaml", "yaml", comm);
#endif

    Node table, opts;
    opts["add_rank"] = 1;
    blueprint::mpi::mesh::flatten(mesh, opts, table, comm);

    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    const std::string filename = baseline_file("braid");
    if(rank == 0)
    {
#ifdef GENERATE_BASELINES
        make_baseline(filename, table);
#endif
        Node baseline;
        load_baseline(filename, baseline);
        table::compare_to_baseline(table, baseline);
    }
}

TEST(t_blueprint_mpi_mesh_flatten, spiral)
{
    const MPI_Comm comm = MPI_COMM_WORLD;
    Node mesh;
    blueprint::mpi::mesh::examples::spiral_round_robin(4, mesh, comm);

#ifdef WRITE_MESH
    relay::mpi::io::blueprint::save_mesh(mesh, "flatten_spiral.yaml", "yaml", comm);
#endif

    Node table, opts;
    opts["add_rank"] = 1;
    blueprint::mpi::mesh::flatten(mesh, opts, table, comm);

    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    const std::string filename = baseline_file("spiral");
    if(rank == 0)
    {
#ifdef GENERATE_BASELINES
        make_baseline(filename, table);
#endif
        Node baseline;
        load_baseline(filename, baseline);
        table::compare_to_baseline(table, baseline);
    }
}

TEST(t_blueprint_mpi_mesh_flatten, spiral_select_fields)
{
    const MPI_Comm comm = MPI_COMM_WORLD;
    Node mesh;
    blueprint::mpi::mesh::examples::spiral_round_robin(4, mesh, comm);

    // Only want the "rank" field and vertex centers
    //  in the output
    Node table, opts;
    opts["add_rank"].set(0);
    opts["field_names"].append().set("rank");
    opts["add_element_centers"].set(0);
    blueprint::mpi::mesh::flatten(mesh, opts, table, comm);

    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    const std::string filename = baseline_file("spiral_select_fields");
    if(rank == 0)
    {
#ifdef GENERATE_BASELINES
        ASSERT_FALSE(table.has_child("element_data"));
        ASSERT_TRUE(table.has_path("vertex_data/values/rank"));
        make_baseline(filename, table);
#endif
        Node baseline;
        load_baseline(filename, baseline);
        table::compare_to_baseline(table, baseline);
    }
}

TEST(t_blueprint_mpi_mesh_flatten, spiral_missing_fields)
{
    const MPI_Comm comm = MPI_COMM_WORLD;
    Node mesh;
    blueprint::mpi::mesh::examples::spiral_round_robin(6, mesh, comm);


    int rank = -1;
    MPI_Comm_rank(comm, &rank);
    if(rank == 3)
    {
        // Rank 3 will be missing a field
        mesh[0]["fields"].remove_child("rank");
    }

    // Only want the "rank" field and vertex centers
    //  in the output
    Node table, opts;
    opts["add_rank"].set(0);
    opts["field_names"].append().set("rank");
    opts["add_element_centers"].set(0);
    opts["fill_value"].set(42);
    blueprint::mpi::mesh::flatten(mesh, opts, table, comm);

    const std::string filename = baseline_file("spiral_missing_fields");
    if(rank == 0)
    {
#ifdef GENERATE_BASELINES
        ASSERT_FALSE(table.has_child("element_data"));
        ASSERT_TRUE(table.has_path("vertex_data/values/rank"));

        // Rank 3's data should be at the end, and should be the fill_value
        Node &n = table["vertex_data/values/rank"];
        const index_t nrows = n.dtype().number_of_elements();
        Node temp;
        temp.set_external(DataType(n.dtype().id(), 1), n.element_ptr(nrows-1));
        ASSERT_DOUBLE_EQ(42., temp.to_double());
        make_baseline(filename, table);
#endif
        Node baseline;
        load_baseline(filename, baseline);
        table::compare_to_baseline(table, baseline);
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
