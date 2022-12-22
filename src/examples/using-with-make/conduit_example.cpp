// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: example.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_blueprint.hpp"


int main(int argc, char **argv)
{
    conduit::Node about;
    conduit::about(about["conduit"]);
    conduit::relay::about(about["conduit/relay"]);
    conduit::relay::io::about(about["conduit/relay/io"]);
    conduit::blueprint::about(about["conduit/blueprint"]);

    std::cout << about.to_yaml() << std::endl;
}

