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
    std::cout << conduit::about() << std::endl
              << conduit::relay::about() << std::endl
              << conduit::blueprint::about() << std::endl;
}


