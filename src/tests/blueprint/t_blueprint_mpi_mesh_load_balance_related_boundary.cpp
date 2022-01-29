// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_parmetis.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
#include "conduit_blueprint_mpi_mesh_parmetis.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_relay_mpi_io.hpp"
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_utils.hpp"
#include "conduit_fmt/conduit_fmt.h"

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;
using namespace conduit::relay::mpi;
using namespace conduit::utils;

using namespace std;


//-----------------------------------------------------------------------------
bool
check_if_hdf5_enabled()
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    return io_protos["io/protocols/hdf5"].as_string() == "enabled";
}


//-----------------------------------------------------------------------------
TEST(blueprint_mpi_load_bal, basic)
{


    //

    int par_size, par_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);

    // build test using related_boundary example
    // one way to create unbalanced number of eles across ranks:
    //  domain 0 --> rank 0
    //  domain 1 and 2 --> rank 0

    Node mesh;
    index_t base_grid_ele_i = 3;
    index_t base_grid_ele_j = 3;

    if(par_rank == 0)
    {
        conduit::blueprint::mesh::examples::related_boundary(base_grid_ele_i,
                                                             base_grid_ele_j,
                                                             mesh);
    }// end par_rank - 0

    std::string output_base = "tout_bp_mpi_load_bal_basic";

    // prefer hdf5, fall back to yaml
    std::string protocol = "yaml";

    if(check_if_hdf5_enabled())
    {
        protocol = "hdf5";
    }

    conduit::relay::mpi::io::blueprint::save_mesh(mesh,
                                                  output_base,
                                                  protocol,
                                                  MPI_COMM_WORLD);
    EXPECT_TRUE(true);
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
