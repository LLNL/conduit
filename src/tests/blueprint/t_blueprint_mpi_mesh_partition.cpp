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
#define GENERATE_BASELINES

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

    // Goes from 7 to 10 domains
    const char *opt0 =
"target: 10";
    options.reset(); options.parse(opt0, "yaml");
    conduit::blueprint::mpi::mesh::partition(input, options, output, MPI_COMM_WORLD);
    if(conduit::blueprint::mesh::number_of_domains(output) > 0)
    {
        std::string b00 = baseline_file((base + "_00_") + rank_str(rank));
        save_visit(b00, output);
#ifdef GENERATE_BASELINES
        make_baseline(b00, output);
#else
        EXPECT_EQ(compare_baseline(b00, output), true);
#endif
    }

    // To do target 2, we may have to store data from 2 ranks.
}

#if 0
//-----------------------------------------------------------------------------
TEST(blueprint_mesh_mpi_partition, split_single_to_multiple)
{
    // Make single domain on rank 0 and then split across 4 ranks into N targets.
}

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
