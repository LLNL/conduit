// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_io_silo.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_io_silo.hpp"
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;


TEST(conduit_relay_io_silo, load_mesh_geometry)
{
    Node mesh, info;
    relay::io::silo::load_mesh("box2d.silo", mesh);

    EXPECT_TRUE(blueprint::mesh::verify(mesh, info));
    EXPECT_EQ(blueprint::mesh::number_of_domains(mesh), 1);

    const Node &domain = *blueprint::mesh::domains(mesh).front();
    EXPECT_TRUE(domain.has_child("coordsets"));
    EXPECT_EQ(domain["coordsets"].number_of_children(), 1);
    EXPECT_TRUE(domain.has_child("topologies"));
    EXPECT_EQ(domain["topologies"].number_of_children(), 1);

    { // Coordset Validation //
        const Node &cset = domain["coordsets"].child(0);

        EXPECT_EQ(blueprint::mesh::coordset::dims(cset, 2));
        EXPECT_EQ(blueprint::mesh::coordset::length(cset, 4));
        EXPECT_TRUE(blueprint::mesh::coordset::_explicit::verify(cset, info));
    }

    { // Topology Validation //
        const Node &topo = domain["topologies"].child(0);
        EXPECT_EQ(blueprint::mesh::topology::dims(cset, 2));
        EXPECT_EQ(blueprint::mesh::topology::length(cset, 1));
        EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(cset, info));
    }
}


TEST(conduit_relay_io_silo, save_mesh_geometry)
{
    Node save_mesh;
    blueprint::mesh::examples::basic("quads", 2, 2, 0, save_mesh);
    save_mesh.remove("fields");
    relay::io::silo::save_mesh(save_mesh, "basic.silo");

    Node load_mesh;
    relay::io::silo::load_mesh("basic.silo", load_mesh);

    Node info;
    EXPECT_FALSE(load_mesh.diff(save_mesh, info));
}
