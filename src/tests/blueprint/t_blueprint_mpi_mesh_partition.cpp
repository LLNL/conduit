// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_mpi_partition.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_blueprint_mpi_mesh.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <math.h>
#include <iostream>
#include "gtest/gtest.h"

#include <mpi.h>

using std::cout;
using std::endl;

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
std::string test_name() { return std::string("t_blueprint_mpi_mesh_partition"); }

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

//-----------------------------------------------------------------------------
void
make_offsets(const int ndomains[4], int offsets[4])
{
    offsets[0] = 0;
    for(int i = 1; i < 4; i++)
        offsets[i] = offsets[i-1] + ndomains[i-1];
}

//-----------------------------------------------------------------------------
void
distribute_domains(int rank, const int ndomains[4],
    const conduit::Node &src_domains, conduit::Node &domains)
{
    int offsets[4];
    make_offsets(ndomains, offsets);

    // Limit to just the domains for this rank.
    domains.reset();
    for(size_t i = 0; i < ndomains[rank]; i++)
    {
        conduit::Node &dom = domains.append();
        dom.set_external(src_domains[offsets[rank] + i]);
    }
}

//-----------------------------------------------------------------------------
std::string
rank_str(int rank)
{
    char tmp[20];
    sprintf(tmp, "%02d", rank);
    return std::string(tmp);
}

//-----------------------------------------------------------------------------
TEST(blueprint_mesh_mpi_partition, all_ranks)
{
    const std::string base("all_ranks");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Make some domains that we'll distribute in different ways.
    conduit::Node spiral;
    conduit::blueprint::mesh::examples::spiral(7, spiral);
#if 1
    if(rank == 0)
        save_visit("spiral", spiral);
#endif
    conduit::Node input, output, options;
    int ndomains[] = {3,2,1,1};
    distribute_domains(rank, ndomains, spiral, input);
#if 0
    // Go from 7 to 10 domains
    const char *opt0 =
"target: 10";
    options.reset(); options.parse(opt0, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b00 = baseline_file((base + "_00_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b00, output);
#ifdef GENERATE_BASELINES
        make_baseline(b00, output);
#else
        EXPECT_EQ(compare_baseline(b00, output), true);
#endif
    }

    // Go from 7 to 4 domains
    const char *opt1 =
"target: 4";
    options.reset(); options.parse(opt1, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b01 = baseline_file((base + "_01_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b01, output);
#ifdef GENERATE_BASELINES
        make_baseline(b01, output);
#else
        EXPECT_EQ(compare_baseline(b01, output), true);
#endif
    }

    // Go from 7 to 2 domains (some ranks donate data but end up with none)
    const char *opt2 =
"target: 2";
    options.reset(); options.parse(opt2, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b02 = baseline_file((base + "_02_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b02, output);
#ifdef GENERATE_BASELINES
        make_baseline(b02, output);
#else
        EXPECT_EQ(compare_baseline(b02, output), true);
#endif
    }

    // Go from 7 to 1 domains
    const char *opt3 =
"target: 1";
    options.reset(); options.parse(opt3, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b03 = baseline_file((base + "_03_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b03, output);
#ifdef GENERATE_BASELINES
        make_baseline(b03, output);
#else
        EXPECT_EQ(compare_baseline(b03, output), true);
#endif
    }
#endif
    // Select an IJK subset of the meshes. Not all ranks will have data for each selection.
    const char *opt4 =
"selections:\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [0,0,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 1\n"
"     start: [0,0,0]\n"
"     end:   [0,0,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 2\n"
"     start: [0,0,0]\n"
"     end:   [1,1,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 3\n"
"     start: [1,0,0]\n"
"     end:   [2,2,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 4\n"
"     start: [1,1,0]\n"
"     end:   [5,5,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 5\n"
"     start: [0,1,0]\n"
"     end:   [3,7,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 6\n"
"     start: [1,0,0]\n"
"     end:   [8,8,0]\n";
    options.reset(); options.parse(opt4, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b04 = baseline_file((base + "_04_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b04, output);
#ifdef GENERATE_BASELINES
        make_baseline(b04, output);
#else
        EXPECT_EQ(compare_baseline(b04, output), true);
#endif
    }

    // Select an IJK subset of the meshes. The last 2 ranks will not make any
    // selections since their domains are not selected.
    const char *opt5 =
"selections:\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [0,0,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 1\n"
"     start: [0,0,0]\n"
"     end:   [0,0,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 2\n"
"     start: [0,0,0]\n"
"     end:   [1,1,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 3\n"
"     start: [1,0,0]\n"
"     end:   [2,2,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 4\n"
"     start: [1,1,0]\n"
"     end:   [5,5,0]\n";
    options.reset(); options.parse(opt5, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b05 = baseline_file((base + "_05_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b05, output);
#ifdef GENERATE_BASELINES
        make_baseline(b05, output);
#else
        EXPECT_EQ(compare_baseline(b05, output), true);
#endif
    }

    // Combine the subselected mesh
    std::string opt6(opt5);
    opt6 += "target: 2";
    options.reset(); options.parse(opt6, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b06 = baseline_file((base + "_06_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b06, output);
#ifdef GENERATE_BASELINES
        make_baseline(b06, output);
#else
        EXPECT_EQ(compare_baseline(b06, output), true);
#endif
    }
/*
ERROR:

Some domains passed through whole so they do not have original cells/domains. Then
when we combine, those fields get skipped because they are not present on all domains.

[/media/bjw/Development/LLNL-Conduit/conduit/src/libs/blueprint/conduit_blueprint_mesh_partition.cpp : 6550]
 Field original_vertex_ids is not present on all input domains, skipping...
[/media/bjw/Development/LLNL-Conduit/conduit/src/libs/blueprint/conduit_blueprint_mesh_partition.cpp : 6550]
 Field original_element_ids is not present on all input domains, skipping...
[/media/bjw/Development/LLNL-Conduit/conduit/src/libs/blueprint/conduit_blueprint_mesh_partition.cpp : 6550]
 Field original_vertex_ids is not present on all input domains, skipping...
[/media/bjw/Development/LLNL-Conduit/conduit/src/libs/blueprint/conduit_blueprint_mesh_partition.cpp : 6550]
 Field original_element_ids is not present on all input domains, skipping...
*/

}

//-----------------------------------------------------------------------------
TEST(blueprint_mesh_mpi_partition, split_single)
{
    const std::string base("split_single");

    // Make single domain on rank 0 and then split across 4 ranks into N targets.

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Make 20x20x1 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {21,21,1};
    if(rank == 0)
    {
        conduit::blueprint::mesh::examples::braid("uniform", vdims[0], vdims[1], vdims[2], input);
        // Override with int64 because YAML loses int/uint information.
        conduit::int64 i100 = 100;
        input["state/cycle"].set(i100);
        input["state/domain_id"] = 0;
    }

    // Go from 1 to 2 domains
    const char *opt0 =
"target: 2";
    options.reset(); options.parse(opt0, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b00 = baseline_file((base + "_00_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b00, output);
#ifdef GENERATE_BASELINES
        make_baseline(b00, output);
#else
        EXPECT_EQ(compare_baseline(b00, output), true);
#endif
    }
    
    // Go from 1 to 5 domains
    const char *opt1 =
"target: 5";
    options.reset(); options.parse(opt1, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    std::string b01 = baseline_file((base + "_01_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        save_visit(b01, output);
#ifdef GENERATE_BASELINES
        make_baseline(b01, output);
#else
        EXPECT_EQ(compare_baseline(b01, output), true);
#endif
    }
}

//-----------------------------------------------------------------------------
TEST(blueprint_mesh_mpi_partition, split_combine)
{
    const std::string base("split_combine");

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Make 20x20x1 cell mesh.
    conduit::Node input, output, output2, options, msg;
    conduit::index_t vdims[] = {21,21,1};
    if(rank == 0)
    {
        conduit::blueprint::mesh::examples::braid("uniform", vdims[0], vdims[1], vdims[2], input);
        // Override with int64 because YAML loses int/uint information.
        conduit::int64 i100 = 100;
        input["state/cycle"].set(i100);
        input["state/domain_id"] = 0;
    }

    // Go from 1 to 5 domains then back to 1. Make sure it is still uniform.
    const char *opt0 =
"target: 5";
    options.reset(); options.parse(opt0, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
const char *opt01 =
"target: 1";
    options.reset(); options.parse(opt01, "yaml");
    conduit::blueprint::mpi::mesh::partition(output, options, output2, MPI_COMM_WORLD);
    std::string b00 = baseline_file((base + "_00_") + rank_str(rank));
    if(conduit::blueprint::mesh::number_of_domains(output2) > 0)
    {
        EXPECT_EQ(output2["topologies/mesh/type"].as_string(), "uniform");
        save_visit(b00, output2);
#ifdef GENERATE_BASELINES
        make_baseline(b00, output2);
#else
        EXPECT_EQ(compare_baseline(b00, output2), true);
#endif
    }
}

#if 0
//-----------------------------------------------------------------------------
TEST(blueprint_mesh_mpi_partition, all_ranks_have_data_selections)
{
}

//-----------------------------------------------------------------------------
TEST(blueprint_mesh_mpi_partition, some_ranks_have_data)
{
}

//-----------------------------------------------------------------------------
TEST(blueprint_mesh_mpi_partition, some_ranks_have_data_selectons)
{
}

// NOTE: what about a test that has different selections on each rank?

#endif

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size == 4)
        result = RUN_ALL_TESTS();
    else
    {
        cout << "This program requires 4 ranks." << endl;
        result = -1;
    }
    MPI_Finalize();

    return result;
}
