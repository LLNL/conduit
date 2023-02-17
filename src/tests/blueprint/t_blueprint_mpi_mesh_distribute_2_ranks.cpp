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
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <math.h>
#include <iostream>
#include "gtest/gtest.h"

#include <mpi.h>

using namespace conduit;


//-----------------------------------------------------------------------------
bool
check_if_hdf5_enabled()
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    return io_protos["io/protocols/hdf5"].as_string() == "enabled";
}


//-----------------------------------------------------------------------------
std::string save_protocol()
{
        if(check_if_hdf5_enabled())
        {
            return "hdf5";
        }
        else
        {
            return "yaml";
        }
}

//-----------------------------------------------------------------------------
TEST(ascent_mpi_mfem_dist_example, test_2_ranks)
{
    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    CONDUIT_INFO("Rank "
                 << par_rank
                 << " of "
                 << par_size
                 << " reporting");
    MPI_Barrier(comm);

    //
    // Create the data.
    //
    Node input_full;
    blueprint::mesh::examples::spiral(par_size,input_full);
    
    Node input;
    //
    input.append().set(input_full[par_rank]);
    input[0]["state/domain_id"] = par_rank;

    /////////////////
    /// CASE 0
    /////////////////

    // show input
    if(par_rank == 0)
    {
        std::cout << "Distribute 2 Ranks Case 0 Input" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << input.to_yaml() << std::endl;
    }
    MPI_Barrier(comm);

    if(par_rank == 1) 
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << input.to_yaml() << std::endl;
    }
    MPI_Barrier(comm);

    // save input 
    conduit::relay::mpi::io::blueprint::save_mesh(input,"tout_mpi_dist_case_0_input",save_protocol(),comm);

    Node opts,res;
    opts["domain_map/values"] = {1,0};
    conduit::blueprint::mpi::mesh::distribute(input,opts,res,comm);

    // save output
    conduit::relay::mpi::io::blueprint::save_mesh(res,"tout_mpi_dist_case_0_output",save_protocol(),comm);

    // check domain ids and show result
    if(par_rank == 0)
    {
        std::cout << "Distribute 2 Ranks Case 0 Output" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << res.to_yaml() << std::endl;
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);

    if(par_rank == 1) 
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << res.to_yaml() << std::endl;
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    }
    MPI_Barrier(comm);

    /////////////////
    /// CASE 1
    /////////////////

    // show input
    if(par_rank == 0)
    {
        std::cout << "Distribute 2 Ranks Case 1 Input" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << input.to_yaml() << std::endl;
    }
    MPI_Barrier(comm);

    if(par_rank == 1) 
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << input.to_yaml() << std::endl;
    }
    MPI_Barrier(comm);

    // save input 
    conduit::relay::mpi::io::blueprint::save_mesh(input,"tout_mpi_dist_case_1_input",save_protocol(),comm);

    opts.reset();
    opts["domain_map/values"]  = {0,1,0,1};
    opts["domain_map/sizes"]   = {2,2};
    opts["domain_map/offsets"] = {0,2};
    conduit::blueprint::mpi::mesh::distribute(input,opts,res,comm);

    // save output
    conduit::relay::mpi::io::blueprint::save_mesh(res,"tout_mpi_dist_case_1_output",save_protocol(),comm);

    // check domain ids and show result
    if(par_rank == 0)
    {
        std::cout << "Distribute 2 Ranks Case 1 Output" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << res.to_yaml() << std::endl;
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
        EXPECT_EQ(res[1]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);

    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << res.to_yaml() << std::endl;
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
        EXPECT_EQ(res[1]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);


    /////////////////
    /// CASE 2
    /////////////////

    // single domain, but replicated on both ranks
    if(par_rank > 0)
    {
        input.reset();
    }

    // show input
    if(par_rank == 0)
    {
        std::cout << "Distribute 2 Ranks Case 2 Input" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << input.to_yaml() << std::endl;
    }
    MPI_Barrier(comm);

    if(par_rank == 1) 
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << input.to_yaml() << std::endl;
    }
    MPI_Barrier(comm);

    conduit::relay::mpi::io::blueprint::save_mesh(input,"tout_mpi_dist_case_2_input",save_protocol(),comm);

    opts.reset();
    opts["domain_map/values"]  = {0,1};
    opts["domain_map/sizes"]   = {2};
    opts["domain_map/offsets"] = {0};
    conduit::blueprint::mpi::mesh::distribute(input,opts,res,comm);

    conduit::relay::mpi::io::blueprint::save_mesh(res,"tout_mpi_dist_case_2_output",save_protocol(),comm);

    // check domain ids and show result
    if(par_rank == 0)
    {
        std::cout << "Distribute 2 Ranks Case 2 Output" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << res.to_yaml() << std::endl;
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    }
    MPI_Barrier(comm);

    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        std::cout << res.to_yaml() << std::endl;
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    }
    MPI_Barrier(comm);
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size == 2)
        result = RUN_ALL_TESTS();
    else
    {
        std::cout << "This program requires 2 ranks." << std::endl;
        result = -1;
    }
    MPI_Finalize();

    return result;
}
