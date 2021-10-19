// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_flatten.cpp
///
//-----------------------------------------------------------------------------

#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>

#include <gtest/gtest.h>

using namespace conduit;

TEST(blueprint_mesh_flatten, basic)
{
    Node mesh, opts;
    blueprint::mesh::examples::basic("uniform", 4, 4, 4, mesh);

    Node table;
    blueprint::mesh::flatten(mesh, opts, table);

    std::cout << table.to_json() << std::endl;
}

TEST(blueprint_mesh_flatten, spiral)
{
    Node mesh, opts;
    blueprint::mesh::examples::spiral(3, mesh);
    conduit::relay::io::blueprint::write_mesh(mesh, "spiral_example", "yaml");

    Node table;
    blueprint::mesh::flatten(mesh, opts, table);
    conduit::relay::io::save(table, "spiral_example_table", "yaml");

    std::cout << table.to_json() << std::endl;
}
