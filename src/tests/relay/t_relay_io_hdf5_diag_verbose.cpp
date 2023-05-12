// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_io_hdf5_diag_verbose.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_io_hdf5.hpp"
#include "hdf5.h"
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
TEST(conduit_relay_io_hdf5_diag, diag_verbose_and_quiet)
{
    Node opts;
    opts["messages"] = "verbose";
    io::hdf5_set_options(opts);

    Node n;
    n["my/data"] = 42;

    std::cout << "[verbose mode -- invoke error]" << std::endl;

    EXPECT_THROW(io::hdf5_write(n,"/garbage/path/wont/work"),conduit::Error);

    opts["messages"] = "quiet"; // default mode
    io::hdf5_set_options(opts);

    std::cout << "[quiet mode -- invoke error]" << std::endl;
    EXPECT_THROW(io::hdf5_write(n,"/garbage/path/wont/work"),conduit::Error);

}
