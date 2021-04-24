// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_mpi_example.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_relay_mpi_io.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    //much ado about `about`

    conduit::Node about;
    conduit::about(about["conduit"]);
    conduit::relay::about(about["conduit/relay"]);
    conduit::relay::io::about(about["conduit/relay/io"]);
    conduit::blueprint::about(about["conduit/blueprint"]);

    conduit::Node about_mpi;
    conduit::relay::mpi::about(about_mpi["conduit/relay/mpi"]);
    conduit::relay::mpi::io::about(about_mpi["conduit/relay/mpi/io"],MPI_COMM_WORLD);
    conduit::blueprint::mpi::about(about_mpi["conduit/blueprint/mpi"]);

    std::cout << about.to_yaml() << std::endl;
    std::cout << about_mpi.to_yaml() << std::endl;

    MPI_Finalize();
}


