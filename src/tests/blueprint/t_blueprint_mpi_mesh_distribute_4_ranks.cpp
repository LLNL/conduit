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
TEST(ascent_mpi_mfem_dist_example, test_4_ranks)
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
    input.append().set(input_full[par_rank]);
    input[0]["state/domain_id"] = par_rank;

    conduit::relay::mpi::io::blueprint::save_mesh(input,"tout_mpi_dist_4r_case_0_input","hdf5",comm);

    /////////////////
    /// CASE 0
    /////////////////

    MPI_Barrier(comm);
    if(par_rank == 0)
    {
        std::cout << "Distribute 4 Ranks Case 0 Input" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);

    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);
    
    if(par_rank == 2)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);
    
    if(par_rank == 3)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);

    Node opts,res;
    opts["domain_map/values"] = {3,2,1,0};
    conduit::blueprint::mpi::mesh::distribute(input,opts,res,comm);

    conduit::relay::mpi::io::blueprint::save_mesh(res,"tout_mpi_dist_4r_case_0_output","hdf5",comm);

    if(par_rank == 0)
    {
        std::cout << "Distribute 4 Ranks Case 0 Result" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),3);
    }
    MPI_Barrier(comm);

    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),2);
    }
    MPI_Barrier(comm);
    
    if(par_rank == 2)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);
    
    if(par_rank == 3)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    }
    MPI_Barrier(comm);

    blueprint::mesh::examples::spiral(par_size,input_full);
    input.reset();
    input.append().set(input_full[par_rank]);
    input[0]["state/domain_id"] = par_rank;

    /////////////////
    /// CASE 1
    /////////////////

    // only first two ranks have data
    if(par_rank >= 2)
    {
        input.reset();
    }

    if(par_rank == 0)
    {
        std::cout << "Distribute 4 Ranks Case 1 Input" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);

    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);
    
    if(par_rank == 2)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);
    
    if(par_rank == 3)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);

    conduit::relay::mpi::io::blueprint::save_mesh(input,"tout_mpi_dist_4r_case_1_input","hdf5",comm);

    opts.reset();
    opts["domain_map/values"]  = {1,2,3, 0,1};
    opts["domain_map/sizes"]   = {3, 2};
    opts["domain_map/offsets"] = {0, 3};
    conduit::blueprint::mpi::mesh::distribute(input,opts,res,comm);


    conduit::relay::mpi::io::blueprint::save_mesh(res,"tout_mpi_dist_4r_case_1_output","hdf5",comm);

    if(par_rank == 0)
    {
        std::cout << "Distribute 4 Ranks Case 1 Output" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);
    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
        EXPECT_EQ(res[1]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);
    if(par_rank == 2)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    }
    MPI_Barrier(comm);
    if(par_rank == 3)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    }
    MPI_Barrier(comm);

    /////////////////
    /// CASE 2
    /////////////////

    // same as case 1, but we expect offsets to be auto generated
    if(par_rank == 0)
    {
        std::cout << "Distribute 4 Ranks Case 1 Input" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);

    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);
    
    if(par_rank == 2)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);
    
    if(par_rank == 3)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        input.print();
    }
    MPI_Barrier(comm);

    conduit::relay::mpi::io::blueprint::save_mesh(input,"tout_mpi_dist_4r_case_1_input","hdf5",comm);

    opts.reset();
    opts["domain_map/values"]  = {1,2,3, 0,1};
    opts["domain_map/sizes"]   = {3, 2};
    // same as case 1, but we expect offsets to be auto generated
    conduit::blueprint::mpi::mesh::distribute(input,opts,res,comm);


    conduit::relay::mpi::io::blueprint::save_mesh(res,"tout_mpi_dist_4r_case_1_output","hdf5",comm);

    if(par_rank == 0)
    {
        std::cout << "Distribute 4 Ranks Case 2 Output" << std::endl;
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);
    if(par_rank == 1)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
        EXPECT_EQ(res[1]["state/domain_id"].to_index_t(),1);
    }
    MPI_Barrier(comm);
    if(par_rank == 2)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
        EXPECT_EQ(res[0]["state/domain_id"].to_index_t(),0);
    }
    MPI_Barrier(comm);
    if(par_rank == 3)
    {
        std::cout << "[rank " << par_rank << "]" << std::endl;
        res.print();
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
    if(size == 4)
        result = RUN_ALL_TESTS();
    else
    {
        std::cout << "This program requires 4 ranks." << std::endl;
        result = -1;
    }
    MPI_Finalize();

    return result;
}
