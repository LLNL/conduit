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

void
silo_uniform_to_rect_conversion(const std::string &coordset_name,
                                const std::string &topo_name,
                                conduit::Node &save_mesh)
{
    Node save_mesh_rect;
    Node &save_mesh_rect_coords = save_mesh_rect["coordsets"][coordset_name];
    Node &save_mesh_rect_topo = save_mesh_rect["topologies"][topo_name];
    blueprint::mesh::topology::uniform::to_rectilinear(
        save_mesh["topologies"][topo_name], 
        save_mesh_rect_topo, save_mesh_rect_coords);
    save_mesh["topologies"][topo_name].set(save_mesh_rect_topo);
    save_mesh["coordsets"][coordset_name].set(save_mesh_rect_coords);
}

// we assume 1 coordset, 1 topo, and fields
void
silo_name_changer(const std::string &mmesh_name,
                  conduit::Node &save_mesh)
{
    Node &coordsets = save_mesh["coordsets"];
    Node &topologies = save_mesh["topologies"];
    Node &fields = save_mesh["fields"];

    // we assume only 1 child for each
    std::string coordset_name = coordsets.children().next().name();
    std::string topo_name = topologies.children().next().name();

    // come up with new names for coordset and topo
    std::string new_coordset_name = mmesh_name + "_" + topo_name;
    std::string new_topo_name = mmesh_name + "_" + topo_name;

    // rename the coordset and references to it
    coordsets.rename_child(coordset_name, new_coordset_name);
    topologies[topo_name]["coordset"].reset();
    topologies[topo_name]["coordset"] = new_coordset_name;

    // rename the topo
    topologies.rename_child(topo_name, new_topo_name);

    auto field_itr = fields.children();
    while (field_itr.has_next())
    {
        Node &n_field = field_itr.next();
        std::string field_name = field_itr.name();

        // use new topo name
        n_field["topology"].reset();
        n_field["topology"] = new_topo_name;

        // remove vol dep
        if (n_field.has_child("volume_dependent"))
        {
            n_field.remove_child("volume_dependent");
        }

        // we need to rename vector components
        if (n_field["values"].dtype().is_object())
        {
            if (n_field["values"].number_of_children() > 0)
            {
                int child_index = 0;
                auto val_itr = n_field["values"].children();
                while (val_itr.has_next())
                {
                    val_itr.next();
                    std::string comp_name = val_itr.name();

                    // rename vector components
                    n_field["values"].rename_child(comp_name, std::to_string(child_index));

                    child_index ++;
                }
            }
        }

        // come up with new field name
        std::string new_field_name = mmesh_name + "_" + field_name;

        // rename the field
        fields.rename_child(field_name, new_field_name);
    }

    if (!save_mesh.has_path("state/domain_id"))
    {
        save_mesh["state"]["domain_id"] = 0;
    }
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

// test reading in a handful of different silo files
TEST(conduit_relay_io_silo, load_mesh_geometry)
{

    // TODO: all these files are in overlink symlink format.
    // Symlinks may break on Windows (?)
    // Could make them overlink format without the symlink.
    // But would require modifying the files.
    std::vector<std::string> filename_vec = {
        "box2d.silo",
        "box3d.silo",
        // "diamond.silo", <--- this one fails because polytopal is not yet supported
        // TODO: rename these files to be more descriptive.
        // would also require modifying the paths stored within the files,
        // and re-symlinking
        "testDisk2D_a.silo",
        // "donordiv.s2_materials2.silo",
        "donordiv.s2_materials3.silo"
    };
    std::vector<int> dims_vec            = {2, 3, /*2,*/  2,    /*2,*/  2};
    std::vector<int> coordset_length_vec = {4, 8, /*36,*/ 1994, /*16,*/ 961};
    std::vector<int> topology_length_vec = {1, 1, /*33,*/ 1920, /*9,*/  900};
    for (int i = 0; i < filename_vec.size(); ++i) 
    {
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


// TODO: add tests for matsets
TEST(conduit_relay_io_silo, save_mesh_geometry_basic)
{
    const std::vector<std::pair<std::string, int>> mesh_types = {
        std::make_pair("uniform", 2), std::make_pair("uniform", 3),
        std::make_pair("rectilinear", 2), std::make_pair("rectilinear", 3),
        std::make_pair("structured", 2), std::make_pair("structured", 3),
        std::make_pair("tris", 2),
        std::make_pair("quads", 2),
        // std::make_pair("polygons", 2),
        std::make_pair("tets", 3),
        std::make_pair("hexs", 3),
        std::make_pair("wedges", 3),
        std::make_pair("pyramids", 3),
        // std::make_pair("polyhedra", 3)
    };
    for (int i = 0; i < mesh_types.size(); ++i)
    {
        int dim = mesh_types[i].second;
        index_t nx = 3;
        index_t ny = 4;
        index_t nz = (dim == 2 ? 0 : 2);

        std::string mesh_type = mesh_types[i].first;

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::basic(mesh_type, nx, ny, nz, save_mesh);
        io::silo::save_mesh(save_mesh, "basic");
        io::silo::load_mesh("basic.root", load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        // check that load mesh correctly adds the state
        save_mesh["state/cycle"] = 0;
        save_mesh["state/domain_id"] = 0;
        // The silo conversion will transform uniform to rectilinear
        // so we will do the same to allow the diff to succeed
        if (mesh_type == "uniform")
            silo_uniform_to_rect_conversion("coords", "mesh", save_mesh);

        // The Blueprint to Silo transformation changes several names 
        // and some information is lost. We manually make changes so 
        // that the diff will pass.
        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // (it will be a list containing a single mesh domain)
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
    }
}

// we are testing vector fields in this test
TEST(conduit_relay_io_silo, save_mesh_geometry_braid)
{
    const std::vector<std::pair<std::string, int>> mesh_types = {
        std::make_pair("uniform", 2), std::make_pair("uniform", 3),
        std::make_pair("rectilinear", 2), std::make_pair("rectilinear", 3),
        std::make_pair("structured", 2), std::make_pair("structured", 3),
        // std::make_pair("point", 2), std::make_pair("point", 3), // TODO
        std::make_pair("lines", 2), std::make_pair("lines", 3),
        std::make_pair("tris", 2),
        std::make_pair("quads", 2),
        std::make_pair("tets", 3),
        std::make_pair("hexs", 3),
        std::make_pair("wedges", 3),
        std::make_pair("pyramids", 3),
        // std::make_pair("mixed_2d", 2),
        // std::make_pair("mixed", 3),
    };
    for (int i = 0; i < mesh_types.size(); ++i)
    {
        int dim = mesh_types[i].second;
        index_t nx = 3;
        index_t ny = 4;
        index_t nz = (dim == 2 ? 0 : 2);

        std::string mesh_type = mesh_types[i].first;

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::braid(mesh_type, nx, ny, nz, save_mesh);

        io::silo::save_mesh(save_mesh, "braid");
        io::silo::load_mesh("braid.cycle_000100.root", load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        // The silo conversion will transform uniform to rectilinear
        // so we will do the same to allow the diff to succeed
        if (mesh_type == "uniform")
        {
            silo_uniform_to_rect_conversion("coords", "mesh", save_mesh);
        }

        // The Blueprint to Silo transformation changes several names 
        // and some information is lost. We manually make changes so 
        // that the diff will pass.
        silo_name_changer("mesh", save_mesh);

        // silo will store this value as an int32. For whatever reason,
        // braid stores cycle as a uint64, unlike the other mesh blueprint
        // examples. We must change this so the diff will pass.
        int cycle = save_mesh["state"]["cycle"].as_uint64();
        save_mesh["state"]["cycle"].reset();
        save_mesh["state"]["cycle"] = (int32) cycle;

        // the loaded mesh will be in the multidomain format
        // (it will be a list containing a single mesh domain)
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
    }
}

// multidomain test
TEST(conduit_relay_io_silo, save_mesh_geometry_spiral)
{
    for (int ndomains = 2; ndomains < 6; ndomains ++)
    {
        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::spiral(ndomains, save_mesh);
        io::silo::save_mesh(save_mesh, "spiral");
        io::silo::load_mesh("spiral.cycle_000000.root", load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

        for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
        {
            // The Blueprint to Silo transformation changes several names 
            // and some information is lost. We manually make changes so 
            // that the diff will pass.
            silo_name_changer("mesh", save_mesh[child]);
        }

        EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
        NodeConstIterator l_itr = load_mesh.children();
        NodeConstIterator s_itr = save_mesh.children();
        while (l_itr.has_next())
        {
            const Node &l_curr = l_itr.next();
            const Node &s_curr = s_itr.next();

            EXPECT_FALSE(l_curr.diff(s_curr, info));
        }
    }
}

// TODO units?
