// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_table_relay.cpp
///
//-----------------------------------------------------------------------------

#include <conduit.hpp>
#include <conduit_blueprint_mesh.hpp>
#include <conduit_blueprint_mesh_examples.hpp>
#include <conduit_blueprint_table_examples.hpp>
#include <conduit_relay_io.hpp>
#include <conduit_relay_io_csv.hpp>

#include "gtest/gtest.h"

using namespace conduit;

TEST(t_blueprint_table_relay, basic_table)
{
    Node table;
    blueprint::table::examples::basic(5, 4, 3, table);
    Node opts;
    relay::io::write_csv(table, "basic.csv", opts);
}
