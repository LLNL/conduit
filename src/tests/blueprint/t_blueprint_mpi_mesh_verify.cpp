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

#include "conduit_fmt/conduit_fmt.h"

#include <iostream>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_smoke, basic_verify)
{
    conduit::Node mesh, info;

    int par_size;
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);

    // empty on all domains should return false ... 
    EXPECT_FALSE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));
        
    conduit::blueprint::mesh::examples::braid("uniform",
                                      10,
                                      10,
                                      10,
                                      mesh);

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));
    EXPECT_EQ(conduit::blueprint::mpi::mesh::number_of_domains(mesh,MPI_COMM_WORLD),par_size);

    conduit::Node domain_to_rank_map;
    conduit::blueprint::mpi::mesh::generate_domain_to_rank_map(mesh,domain_to_rank_map,MPI_COMM_WORLD);

    EXPECT_TRUE(domain_to_rank_map.dtype().is_int64());

    EXPECT_EQ(domain_to_rank_map.dtype().number_of_elements(),par_size);

    conduit::int64_array map_array = domain_to_rank_map.as_int64_array();
    for(conduit::int64 rank=0; rank < par_size; ++rank)
    {
        EXPECT_EQ(map_array[rank], rank);
    }

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

        // all ranks should have index data.
        EXPECT_TRUE(bp_index["mesh"].dtype().is_object());

    }
}

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_smoke, multi_domain)
{
    conduit::Node mesh, info;

    // empty on all domains should return false ... 
    EXPECT_FALSE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

    int par_rank;
    int par_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);

    // This tests a parallel multi-domain mesh with varying number of domains
    // on each rank.
    //
    // Each rank will have a number of domains equal to (rank%3)+1.

    conduit::index_t start_dom = 0;
    for(conduit::index_t rank=0; rank < par_rank; ++rank)
    {
        start_dom += (rank%3) + 1;
    }

    int num_doms_on_rank = (par_rank%3) + 1;
    int stop_dom = start_dom + num_doms_on_rank;
    for(conduit::index_t dom=start_dom; dom < stop_dom; ++dom)
    {
        std::string domain_name = conduit_fmt::format("domain_{:d}", dom);

        conduit::Node& domain = mesh[domain_name];

        conduit::blueprint::mesh::examples::braid("uniform",
                                                  10,
                                                  10,
                                                  10,
                                                  domain);

        domain["state/domain_id"] = dom;

    }

    EXPECT_TRUE( conduit::blueprint::mpi::verify("mesh",mesh,info, MPI_COMM_WORLD));

    // check the total number of domains.
    conduit::index_t num_domains = 0;
    for(conduit::index_t rank=0; rank < par_size; ++rank)
    {
        num_domains += (rank%3) + 1;
    }

    EXPECT_EQ( conduit::blueprint::mpi::mesh::number_of_domains(mesh,MPI_COMM_WORLD), num_domains);

    // check hypothetical index gen
    conduit::Node bp_index;
    conduit::blueprint::mpi::mesh::generate_index(mesh,
                                                  "",
                                                  bp_index["mesh"],
                                                  MPI_COMM_WORLD);

    // all ranks should have index data.
    EXPECT_TRUE(bp_index["mesh"].dtype().is_object());

    conduit::Node domain_to_rank_map;
    conduit::blueprint::mpi::mesh::generate_domain_to_rank_map(mesh,domain_to_rank_map,MPI_COMM_WORLD);

    EXPECT_TRUE(domain_to_rank_map.dtype().is_int64());

    EXPECT_EQ(domain_to_rank_map.dtype().number_of_elements(), num_domains);

    // Check the values in domain_to_rank_map.
    conduit::int64_array map_array = domain_to_rank_map.as_int64_array();
    conduit::int64 start_dom_for_rank = 0;
    for(conduit::int64 cur_rank=0; cur_rank < par_size; ++cur_rank)
    {
        conduit::int64 stop_dom_for_rank = start_dom_for_rank + (cur_rank%3)+1;
        for(conduit::int64 dom=start_dom_for_rank; dom < stop_dom_for_rank;
            ++dom)
        {
            EXPECT_EQ(map_array[dom], cur_rank);
        }

        start_dom_for_rank += (cur_rank%3) + 1;
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
