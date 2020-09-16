// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_verify.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"

#include <iostream>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_smoke, basic_verify)
{
    conduit::Node mesh, info;

    // empty on all domains should return false ... 
    EXPECT_FALSE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));
        
    conduit::blueprint::mesh::examples::braid("uniform",
                                      10,
                                      10,
                                      10,
                                      mesh);

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));
    EXPECT_EQ(conduit::blueprint::mpi::mesh::number_of_domains(mesh,MPI_COMM_WORLD),2);
}


//-----------------------------------------------------------------------------
TEST(blueprint_mpi_smoke, ranks_with_no_mesh)
{
    conduit::Node mesh, info;

    // empty on all domains should return false ... 
    EXPECT_FALSE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

    int par_rank;
    int par_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    
    for(conduit::index_t active_rank=0;active_rank < par_size; active_rank++)
    {
        mesh.reset();
        // even with a single domain on one rank, we should still verify true
        if(par_rank == active_rank)
        {
            conduit::blueprint::mesh::examples::braid("uniform",
                                                      10,
                                                      10,
                                                      10,
                                                      mesh);
        }

        EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

        // check the number of domains
        EXPECT_EQ( conduit::blueprint::mpi::mesh::number_of_domains(mesh,MPI_COMM_WORLD), 1);

        // check hypothetical index gen
        conduit::Node bp_index;
        conduit::blueprint::mpi::mesh::generate_index(mesh,
                                                      "",
                                                      bp_index["mesh"],
                                                      MPI_COMM_WORLD);
    }
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
