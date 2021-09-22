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
#include "conduit_blueprint.hpp"
#include "t_config.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;


std::string
relay_test_silo_data_path(const std::string &test_fname)
{
    std::string res = utils::join_path(CONDUIT_T_SRC_DIR, "relay");
    res = utils::join_file_path(res, "data");
    res = utils::join_file_path(res, "silo");
    return utils::join_file_path(res, test_fname);
}


TEST(conduit_relay_io_silo, conduit_silo_cold_storage)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);

    io::silo_write(n,"tout_cold_storage_test.silo:myobj");

    Node n_load;
    io::silo_read("tout_cold_storage_test.silo:myobj",n_load);

    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
    EXPECT_EQ(n_load["c"].as_uint32(), c_val);
}

TEST(conduit_relay_io_silo, conduit_silo_cold_storage_generic_iface)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;

    EXPECT_EQ(n["a"].as_uint32(), a_val);
    EXPECT_EQ(n["b"].as_uint32(), b_val);
    EXPECT_EQ(n["c"].as_uint32(), c_val);

    io::save(n, "tout_cold_storage_test_generic_iface.silo:myobj");

    Node n_load;
    io::load("tout_cold_storage_test_generic_iface.silo:myobj",n_load);

    EXPECT_EQ(n_load["a"].as_uint32(), a_val);
    EXPECT_EQ(n_load["b"].as_uint32(), b_val);
    EXPECT_EQ(n_load["c"].as_uint32(), c_val);
}

// test simple silo 2D and 3D boxes
TEST(conduit_relay_io_silo, load_mesh_geometry)
{
    std::vector<std::string> filename_vec = {"box2d.silo", "box3d.silo", "diamond.silo"};
    std::vector<int> dims_vec = {2, 3, 2};
    std::vector<int> coordset_length_vec = {4, 8, 36};
    std::vector<int> topology_length_vec = {1, 1, 33};
    for (int i = 0; i < filename_vec.size(); ++i) {

        Node mesh, info;
        std::string input_file = relay_test_silo_data_path(filename_vec.at(i));
        io::silo::load_mesh(input_file, mesh);

        EXPECT_TRUE(blueprint::mesh::verify(mesh, info));
        EXPECT_EQ(blueprint::mesh::number_of_domains(mesh), 1);

        const Node &domain = *blueprint::mesh::domains(mesh).front();
        EXPECT_TRUE(domain.has_child("coordsets"));
        EXPECT_EQ(domain["coordsets"].number_of_children(), 1);
        EXPECT_TRUE(domain.has_child("topologies"));
        EXPECT_EQ(domain["topologies"].number_of_children(), 1);

        { // Coordset Validation //
            const Node &cset = domain["coordsets"].child(0);

            EXPECT_EQ(blueprint::mesh::coordset::dims(cset), dims_vec.at(i));
            EXPECT_EQ(blueprint::mesh::coordset::length(cset), coordset_length_vec.at(i));
            EXPECT_TRUE(blueprint::mesh::coordset::_explicit::verify(cset, info));
        }

        { // Topology Validation //
            const Node &topo = domain["topologies"].child(0);
            EXPECT_EQ(blueprint::mesh::topology::dims(topo), dims_vec.at(i));
            EXPECT_EQ(blueprint::mesh::topology::length(topo), topology_length_vec.at(i));
            EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(topo, info));
        }
    }
}

TEST(conduit_relay_io_silo, save_mesh_geometry)
{
    Node save_mesh;
    blueprint::mesh::examples::basic("quads", 2, 2, 0, save_mesh);
    save_mesh.remove("fields");
    io::silo::save_mesh(save_mesh, "basic.silo");
    Node load_mesh;
    io::silo::load_mesh("basic.silo", load_mesh);
    Node info;
    EXPECT_FALSE(load_mesh.diff(save_mesh, info));
}
