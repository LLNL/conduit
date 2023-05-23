// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_io_silo.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_io_silo.hpp"
#include "conduit_blueprint.hpp"
#include "t_config.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
using namespace conduit::relay;

//-----------------------------------------------------------------------------
std::string
relay_test_silo_data_path(const std::string &test_fname)
{
    std::string res = utils::join_path(CONDUIT_T_SRC_DIR, "relay");
    res = utils::join_file_path(res, "data");
    res = utils::join_file_path(res, "silo");
    return utils::join_file_path(res, test_fname);
}

//-----------------------------------------------------------------------------
// The Blueprint to Silo to Blueprint round trip will
// transform uniform to rectilinear so we will do the same
// to allow diffs to succeed
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

//-----------------------------------------------------------------------------
// The Blueprint to Silo transformation changes several names 
// and some information is lost. We manually make changes so 
// that the diff will pass.
void
silo_name_changer(const std::string &mmesh_name,
                  conduit::Node &save_mesh)
{
    std::map<std::string, std::string> old_to_new_names;

    if (save_mesh.has_child("topologies"))
    {
        auto topo_itr = save_mesh["topologies"].children();
        while(topo_itr.has_next())
        {
            Node &n_topo = topo_itr.next();
            std::string topo_name = topo_itr.name();
            std::string new_topo_name = mmesh_name + "_" + topo_name;

            old_to_new_names[topo_name] = new_topo_name;

            std::string coordset_name = n_topo["coordset"].as_string();
            std::string new_coordset_name = mmesh_name + "_" + topo_name;

            // change the coordset this topo refers to
            save_mesh["topologies"][topo_name]["coordset"].reset();
            save_mesh["topologies"][topo_name]["coordset"] = new_coordset_name;

            // change the name of the topo
            save_mesh["topologies"].rename_child(topo_name, new_topo_name);

            // change the name of the coordset
            if (save_mesh.has_path("coordsets/" + coordset_name))
            {
                save_mesh["coordsets"].rename_child(coordset_name, new_coordset_name);
            }
        }
    }

    if (save_mesh.has_child("fields"))
    {
        auto field_itr = save_mesh["fields"].children();
        while (field_itr.has_next())
        {
            Node &n_field = field_itr.next();
            std::string field_name = field_itr.name();

            std::string old_topo_name = n_field["topology"].as_string();

            if (old_to_new_names.find(old_topo_name) == old_to_new_names.end())
            {
                continue;
                // If this is the case, we probably need to delete this field.
                // But our job in this function is just to rename things, so we 
                // will just skip.
            }
            std::string new_topo_name = old_to_new_names[old_topo_name];

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
            save_mesh["fields"].rename_child(field_name, new_field_name);
        }
    }

    if (!save_mesh.has_path("state/domain_id"))
    {
        save_mesh["state"]["domain_id"] = 0;
    }
}

//-----------------------------------------------------------------------------
// The Blueprint to Overlink transformation changes several names 
// and some information is lost. We manually make changes so 
// that the diff will pass.
void
overlink_name_changer(conduit::Node &save_mesh)
{
    // we assume 1 coordset, 1 topo, and fields

    Node &coordsets = save_mesh["coordsets"];
    Node &topologies = save_mesh["topologies"];
    Node &fields = save_mesh["fields"];

    // we assume only 1 child for each
    std::string coordset_name = coordsets.children().next().name();
    std::string topo_name = topologies.children().next().name();

    // rename the coordset and references to it
    coordsets.rename_child(coordset_name, "MMESH");
    topologies[topo_name]["coordset"].reset();
    topologies[topo_name]["coordset"] = "MMESH";

    // rename the topo
    topologies.rename_child(topo_name, "MMESH");

    auto field_itr = fields.children();
    while (field_itr.has_next())
    {
        Node &n_field = field_itr.next();
        std::string field_name = field_itr.name();

        // use new topo name
        n_field["topology"].reset();
        n_field["topology"] = "MMESH";

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
    }

    if (!save_mesh.has_path("state/domain_id"))
    {
        save_mesh["state"]["domain_id"] = 0;
    }
}

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
// test reading in a handful of different overlink files
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
        // "donordiv.s2_materials2.silo", <--- this one fails because polytopal is not yet supported
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

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_basic)
{
    const std::vector<std::pair<std::string, std::string>> mesh_types = {
        std::make_pair("uniform", "2"), std::make_pair("uniform", "3"),
        std::make_pair("rectilinear", "2"), std::make_pair("rectilinear", "3"),
        std::make_pair("structured", "2"), std::make_pair("structured", "3"),
        std::make_pair("tris", "2"),
        std::make_pair("quads", "2"),
        // std::make_pair("polygons", "2"),
        std::make_pair("tets", "3"),
        std::make_pair("hexs", "3"),
        std::make_pair("wedges", "3"),
        std::make_pair("pyramids", "3"),
        // std::make_pair("polyhedra", "3")
    };
    for (int i = 0; i < mesh_types.size(); ++i)
    {
        std::string dim = mesh_types[i].second;
        index_t nx = 3;
        index_t ny = 4;
        index_t nz = (dim == "2" ? 0 : 2);

        std::string mesh_type = mesh_types[i].first;

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::basic(mesh_type, nx, ny, nz, save_mesh);

        const std::string basename = "silo_basic_" + mesh_type + "_" + dim + "D";
        const std::string filename = basename + ".root";

        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename);
        io::silo::load_mesh(filename, load_mesh);

        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        // make changes to save mesh so the diff will pass
        save_mesh["state/cycle"] = (int64) 0;
        save_mesh["state/domain_id"] = 0;
        if (mesh_type == "uniform")
        {
            silo_uniform_to_rect_conversion("coords", "mesh", save_mesh);
        }
        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
    }
}

//-----------------------------------------------------------------------------
// we are testing vector fields in this test
TEST(conduit_relay_io_silo, round_trip_braid)
{
    const std::vector<std::pair<std::string, std::string>> mesh_types = {
        std::make_pair("uniform", "2"), std::make_pair("uniform", "3"),
        std::make_pair("rectilinear", "2"), std::make_pair("rectilinear", "3"),
        std::make_pair("structured", "2"), std::make_pair("structured", "3"),
        std::make_pair("points", "2"), std::make_pair("points", "3"),
        std::make_pair("points_implicit", "2"), std::make_pair("points_implicit", "3"),
        std::make_pair("lines", "2"), std::make_pair("lines", "3"),
        std::make_pair("tris", "2"),
        std::make_pair("quads", "2"),
        std::make_pair("tets", "3"),
        std::make_pair("hexs", "3"),
        std::make_pair("wedges", "3"),
        std::make_pair("pyramids", "3"),
        // std::make_pair("mixed_2d", "2"),
        // std::make_pair("mixed", "3"),
    };
    for (int i = 0; i < mesh_types.size(); ++i)
    {
        std::string dim = mesh_types[i].second;
        index_t nx = 3;
        index_t ny = 4;
        index_t nz = (dim == "2" ? 0 : 2);

        std::string mesh_type = mesh_types[i].first;

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::braid(mesh_type, nx, ny, nz, save_mesh);

        const std::string basename = "silo_braid_" + mesh_type + "_" + dim + "D";
        const std::string filename = basename + ".cycle_000100.root";

        // remove existing root file, directory and any output files
        remove_path_if_exists(filename);

        io::silo::save_mesh(save_mesh, basename);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        // make changes to save mesh so the diff will pass
        if (mesh_type == "uniform")
        {
            silo_uniform_to_rect_conversion("coords", "mesh", save_mesh);
        }
        if (mesh_type == "points")
        {
            // this is custom code for braid
            // We know it is correct because the unstructured points version of braid
            // uses every point in the coordset
            save_mesh["topologies"].remove_child("mesh");
            save_mesh["topologies"]["mesh"]["type"] = "points";
            save_mesh["topologies"]["mesh"]["coordset"] = "coords";
        }
        if (mesh_type == "points_implicit" || mesh_type == "points")
        {
            // the association doesn't matter for point meshes
            // we choose vertex by convention
            save_mesh["fields"]["radial"]["association"].reset();
            save_mesh["fields"]["radial"]["association"] = "vertex";
        }
        silo_name_changer("mesh", save_mesh);
        int cycle = save_mesh["state"]["cycle"].as_uint64();
        save_mesh["state"]["cycle"].reset();
        save_mesh["state"]["cycle"] = (int64) cycle;

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
    }
}

//-----------------------------------------------------------------------------
// multidomain test
TEST(conduit_relay_io_silo, round_trip_spiral)
{
    for (int ndomains = 2; ndomains < 6; ndomains ++)
    {
        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::spiral(ndomains, save_mesh);

        const std::string basename = "silo_spiral_" + std::to_string(ndomains) + "_domains";
        const std::string filename = basename + ".cycle_000000.root";

        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename);
        io::silo::load_mesh(filename, load_mesh);

        EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

        // make changes to save mesh so the diff will pass
        for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
        {
            silo_name_changer("mesh", save_mesh[child]);
            int cycle = save_mesh[child]["state"]["cycle"].as_int32();
            save_mesh[child]["state"]["cycle"].reset();
            save_mesh[child]["state"]["cycle"] = (int64) cycle;
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

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_julia)
{
    Node save_mesh, load_mesh, info;
    blueprint::mesh::examples::julia(5,  // nx
                                     5,  // ny
                                     0,  // x_min
                                     10, // x_max
                                     2,  // y_min
                                     7,  // y_max
                                     3,  // c_re
                                     4,  // c_im
                                     save_mesh);

    const std::string basename = "silo_julia";
    const std::string filename = basename + ".root";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename);
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    // make changes to save mesh so the diff will pass
    save_mesh["state/cycle"] = (int64) 0;
    save_mesh["state/domain_id"] = 0;
    silo_name_changer("mesh", save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
}

//-----------------------------------------------------------------------------
// 
// special case tests
// 

//-----------------------------------------------------------------------------
// var is not defined on a domain
// 
// tests the silo "EMPTY" capability
TEST(conduit_relay_io_silo, missing_domain_var)
{
    Node save_mesh, load_mesh, info;
    const int ndomains = 4;
    blueprint::mesh::examples::spiral(ndomains, save_mesh);

    // remove information for a particular domain
    save_mesh[2]["fields"].remove_child("dist");

    const std::string basename = "silo_missing_domain_var_spiral";
    const std::string filename = basename + ".cycle_000000.root";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename);
    io::silo::load_mesh(filename, load_mesh);

    EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

    // make changes to save mesh so the diff will pass
    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        silo_name_changer("mesh", save_mesh[child]);
        int cycle = save_mesh[child]["state"]["cycle"].as_int32();
        save_mesh[child]["state"]["cycle"].reset();
        save_mesh[child]["state"]["cycle"] = (int64) cycle;
    }
    save_mesh[2].remove_child("fields");

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

//-----------------------------------------------------------------------------
// mesh is not defined on a domain
// 
// This case is much less interesting.
// data passes through the clean mesh filter which
// deletes domains that are missing topos.
// They simply are not part of the mesh and so silo 
// doesn't have to deal with it.
TEST(conduit_relay_io_silo, missing_domain_mesh_trivial)
{
    Node save_mesh, load_mesh, info;
    const int ndomains = 4;
    blueprint::mesh::examples::spiral(ndomains, save_mesh);

    // remove information for a particular domain
    save_mesh[2]["topologies"].remove_child("topo");

    const std::string basename = "silo_missing_domain_mesh_trivial_spiral";
    const std::string filename = basename + ".cycle_000000.root";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename);
    io::silo::load_mesh(filename, load_mesh);

    EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

    // make changes to save mesh so the diff will pass
    save_mesh.remove(2);
    save_mesh.rename_child("domain_000003", "domain_000002");
    save_mesh[2]["state"]["domain_id"].reset();
    save_mesh[2]["state"]["domain_id"] = 2;
    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        silo_name_changer("mesh", save_mesh[child]);
        int cycle = save_mesh[child]["state"]["cycle"].as_int32();
        save_mesh[child]["state"]["cycle"].reset();
        save_mesh[child]["state"]["cycle"] = (int64) cycle;
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

//-----------------------------------------------------------------------------
// mesh is not defined on a domain but there are multiple meshes
TEST(conduit_relay_io_silo, missing_domain_mesh)
{
    Node save_mesh, save_mesh2, load_mesh, load_mesh2, info, opts;
    const int ndomains = 4;
    blueprint::mesh::examples::spiral(ndomains, save_mesh);
    blueprint::mesh::examples::spiral(ndomains, save_mesh2);

    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        save_mesh[child]["coordsets"].rename_child("coords", "coords2");
        save_mesh[child]["topologies"]["topo"]["coordset"].reset();
        save_mesh[child]["topologies"]["topo"]["coordset"] = "coords2";
        save_mesh[child]["topologies"].rename_child("topo", "topo2");
        save_mesh[child]["fields"]["dist"]["topology"].reset();
        save_mesh[child]["fields"]["dist"]["topology"] = "topo2";
        save_mesh[child]["fields"].rename_child("dist", "dist2");

        save_mesh[child]["coordsets"]["coords"].set_external(save_mesh2[child]["coordsets"]["coords"]);
        save_mesh[child]["topologies"]["topo"].set_external(save_mesh2[child]["topologies"]["topo"]);
        save_mesh[child]["fields"]["dist"].set_external(save_mesh2[child]["fields"]["dist"]);
    }

    // remove information for a particular domain
    save_mesh[2]["topologies"].remove_child("topo");

    const std::string basename = "silo_missing_domain_mesh_spiral";
    const std::string filename = basename + ".cycle_000000.root";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename);
    
    opts["mesh_name"] = "mesh_topo2";
    io::silo::load_mesh(filename, opts, load_mesh);
    opts["mesh_name"] = "mesh_topo";
    io::silo::load_mesh(filename, opts, load_mesh2);

    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh2, info));

    // make changes to save mesh so the diff will pass
    save_mesh[2]["coordsets"].remove_child("coords");
    save_mesh[2]["fields"].remove_child("dist");
    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        silo_name_changer("mesh", save_mesh[child]);
        int cycle = save_mesh[child]["state"]["cycle"].as_int32();
        save_mesh[child]["state"]["cycle"].reset();
        save_mesh[child]["state"]["cycle"] = (int64) cycle;
    }

    // we must merge the two meshes in load mesh
    // this is tricky because one is missing a domain
    load_mesh[0]["coordsets"]["mesh_topo"].set_external(load_mesh2[0]["coordsets"]["mesh_topo"]);
    load_mesh[0]["topologies"]["mesh_topo"].set_external(load_mesh2[0]["topologies"]["mesh_topo"]);
    load_mesh[0]["fields"]["mesh_dist"].set_external(load_mesh2[0]["fields"]["mesh_dist"]);
    load_mesh[1]["coordsets"]["mesh_topo"].set_external(load_mesh2[1]["coordsets"]["mesh_topo"]);
    load_mesh[1]["topologies"]["mesh_topo"].set_external(load_mesh2[1]["topologies"]["mesh_topo"]);
    load_mesh[1]["fields"]["mesh_dist"].set_external(load_mesh2[1]["fields"]["mesh_dist"]);
    load_mesh[3]["coordsets"]["mesh_topo"].set_external(load_mesh2[2]["coordsets"]["mesh_topo"]);
    load_mesh[3]["topologies"]["mesh_topo"].set_external(load_mesh2[2]["topologies"]["mesh_topo"]);
    load_mesh[3]["fields"]["mesh_dist"].set_external(load_mesh2[2]["fields"]["mesh_dist"]);

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

// TODO a test where the explicit points (unstructured mesh) do not use every coord

//-----------------------------------------------------------------------------

// 
// save and read option tests
// 

// save options:
/// opts:
///
///      file_style: "default", "root_only", "multi_file", "overlink"
///            when # of domains == 1,  "default"   ==> "root_only"
///            else,                    "default"   ==> "multi_file"
///
///      silo_type: "default", "pdb", "hdf5", "unknown"
///            when the file we are writing to exists, "default" ==> "unknown"
///            else,                                   "default" ==> "hdf5"
///         note: these are additional silo_type options that we could add 
///         support for in the future:
///           "hdf5_sec2", "hdf5_stdio", "hdf5_mpio", "hdf5_mpiposix", "taurus"
///
///      suffix: "default", "cycle", "none"
///            when cycle is present,  "default"   ==> "cycle"
///            else,                   "default"   ==> "none"
///
///      mesh_name:  (used if present, default ==> "mesh")
///
///      ovl_topo_name: (used if present, default ==> "")
///
///      number_of_files:  {# of files}
///            when "multi_file" or "overlink":
///                 <= 0, use # of files == # of domains
///                  > 0, # of files == number_of_files

// read options:
/// opts:
///      mesh_name: "{name}"
///          provide explicit mesh name, for cases where silo data includes
///           more than one mesh.


// TODO
/// opts:
///
///      number_of_files:  {# of files}
///            when "multi_file" or "overlink":
///                 <= 0, use # of files == # of domains
///                  > 0, # of files == number_of_files

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_file_style)
{
    // we will do overlink tests separately
    const std::vector<std::string> file_styles = {"default", "root_only", "multi_file"};
    for (int i = 0; i < file_styles.size(); i ++)
    {
        Node opts;
        opts["file_style"] = file_styles[i];

        const std::string basename = "silo_save_option_file_style_" + file_styles[i] + "_spiral";
        const std::string filename = basename + ".cycle_000000.root";

        for (int ndomains = 1; ndomains < 5; ndomains += 3)
        {
            Node save_mesh, load_mesh, info;
            blueprint::mesh::examples::spiral(ndomains, save_mesh);
            remove_path_if_exists(filename);
            io::silo::save_mesh(save_mesh, basename, opts);
            io::silo::load_mesh(filename, load_mesh);
            EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

            // make changes to save mesh so the diff will pass
            for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
            {
                silo_name_changer("mesh", save_mesh[child]);
                int cycle = save_mesh[child]["state"]["cycle"].as_int32();
                save_mesh[child]["state"]["cycle"].reset();
                save_mesh[child]["state"]["cycle"] = (int64) cycle;
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
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_suffix)
{
    const std::vector<std::string> suffixes = {"default", "default", "cycle", "none"};
    const std::vector<std::string> file_suffixes = {
        "",              // cycle is not present
        ".cycle_000005", // cycle is present
        ".cycle_000005", // cycle is turned on
        "",              // cycle is turned off
    };
    const std::vector<std::string> include_cycle = {"no", "yes", "yes", "yes"};
    for (int i = 0; i < suffixes.size(); i ++)
    {
        Node opts;
        opts["suffix"] = suffixes[i];

        const std::string basename = "silo_save_option_suffix_" + suffixes[i] +
                                     "_" + include_cycle[i] + "_basic";
        const std::string filename = basename + file_suffixes[i] + ".root";

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::basic("rectilinear", 3, 4, 0, save_mesh);

        if (include_cycle[i] == "yes")
        {
            save_mesh["state/cycle"] = (int64) 5;
        }

        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename, opts);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        // this is to pass the diff, as silo will add cycle in if it is not there
        if (include_cycle[i] == "no")
        {
            save_mesh["state/cycle"] = (int64) 0;
        }
        save_mesh["state/domain_id"] = 0;
        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_mesh_name)
{
    const std::string basename = "silo_save_option_mesh_name_basic";
    const std::string filename = basename + ".root";

    Node opts;
    opts["mesh_name"] = "mymesh";

    Node save_mesh, load_mesh, info;
    blueprint::mesh::examples::basic("rectilinear", 3, 4, 0, save_mesh);
    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename, opts);
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    save_mesh["state/cycle"] = (int64) 0;
    save_mesh["state/domain_id"] = 0;
    silo_name_changer("mymesh", save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_read_option_mesh_name)
{
    Node load_mesh, info, opts;
    std::string input_file = relay_test_silo_data_path("multi_curv3d.silo");

    opts["mesh_name"] = "mesh1_dup";

    io::silo::load_mesh(input_file, opts, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    EXPECT_TRUE(load_mesh[0].has_path("topologies/mesh1_dup"));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_silo_type)
{
    const std::vector<std::string> silo_types = {"default", "pdb", "hdf5", "unknown"};
    for (int i = 3; i < silo_types.size(); i ++)
    {
        Node opts;
        opts["silo_type"] = silo_types[i];

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::basic("rectilinear", 3, 4, 0, save_mesh);

        const std::string basename = "silo_save_option_silo_type_" + silo_types[i] + "_basic";
        const std::string filename = basename + ".root";

        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename, opts);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        // this is to pass the diff, as silo will add cycle in if it is not there
        save_mesh["state/cycle"] = (int64) 0;
        save_mesh["state/domain_id"] = 0;
        
        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_overlink)
{
    const std::vector<std::string> ovl_topo_names = {"", "topo"};
    for (int i = 0; i < ovl_topo_names.size(); i ++)
    {
        std::string basename;
        if (ovl_topo_names[i].empty())
        {
            basename = "silo_save_option_overlink_spiral";
        }
        else
        {
            basename = "silo_save_option_overlink_spiral_" + ovl_topo_names[i];
        }
        const std::string filename = basename + "/OvlTop.silo";

        Node opts;
        opts["file_style"] = "overlink";
        opts["ovl_topo_name"] = ovl_topo_names[i];

        int ndomains = 2;

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::spiral(ndomains, save_mesh);
        remove_path_if_exists(basename);
        io::silo::save_mesh(save_mesh, basename, opts);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

        for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
        {
            overlink_name_changer(save_mesh[child]);
            int cycle = save_mesh[child]["state"]["cycle"].as_int32();
            save_mesh[child]["state"]["cycle"].reset();
            save_mesh[child]["state"]["cycle"] = (int64) cycle;
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

//-----------------------------------------------------------------------------

//
// read and write Silo and Overlink tests
//

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, read_silo)
{
    const std::vector<std::vector<std::string>> file_info = {
        {"./",                  "multi_curv3d",           ".silo", ""}, // test default case
        {"./",                  "multi_curv3d",           ".silo", "mesh1"},
        {"./",                  "multi_curv3d",           ".silo", "mesh1_back"},
        {"./",                  "multi_curv3d",           ".silo", "mesh1_dup"},
        {"./",                  "multi_curv3d",           ".silo", "mesh1_front"},
        {"./",                  "multi_curv3d",           ".silo", "mesh1_hidden"},
        {"./",                  "tire",                   ".silo", ""}, // test default case
        {"./",                  "tire",                   ".silo", "tire"},
        {"./",                  "galaxy0000",             ".silo", ""}, // test default case
        {"./",                  "galaxy0000",             ".silo", "StarMesh"},
        {"./",                  "emptydomains",           ".silo", ""}, // test default case
        {"./",                  "emptydomains",           ".silo", "mesh"},
        {"multidir_test_data/", "multidir0000",           ".root", ""}, // test default case
        {"multidir_test_data/", "multidir0000",           ".root", "Mesh"},
        {"./",                  "box2d",                  ".silo", ""}, // test default case
        {"./",                  "box2d",                  ".silo", "MMESH"},
        {"./",                  "box3d",                  ".silo", ""}, // test default case
        {"./",                  "box3d",                  ".silo", "MMESH"},
     // {"./",                  "diamond",                ".silo", ""}, // test default case
     // {"./",                  "diamond",                ".silo", "MMESH"},
        {"./",                  "testDisk2D_a",           ".silo", ""}, // test default case
        {"./",                  "testDisk2D_a",           ".silo", "MMESH"},
     // {"./",                  "donordiv.s2_materials2", ".silo", ""}, // test default case
     // {"./",                  "donordiv.s2_materials2", ".silo", "MMESH"},
        {"./",                  "donordiv.s2_materials3", ".silo", ""}, // test default case
        {"./",                  "donordiv.s2_materials3", ".silo", "MMESH"},
        {"box2d/",              "OvlTop",                 ".silo", "MMESH"},
        // TODO add tests for reading each of the overlink files from ovltop.silo
        // TODO add tests with the bad overlink visit test data
    };

    for (int i = 0; i < file_info.size(); i ++) 
    {
        const std::string dirname  = file_info[i][0];
        const std::string basename = file_info[i][1];
        const std::string fileext  = file_info[i][2];
        const std::string meshname = file_info[i][3];

        Node load_mesh, info, read_opts, write_opts;
        std::string input_file = relay_test_silo_data_path(dirname + basename + fileext);

        read_opts["mesh_name"] = meshname;

        io::silo::load_mesh(input_file, read_opts, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        std::string out_name = basename;
        if (!meshname.empty())
        {
            out_name += "_" + meshname;
        }

        remove_path_if_exists(out_name + "_blueprint");
        io::blueprint::save_mesh(load_mesh, out_name + "_blueprint", "hdf5");

        remove_path_if_exists(out_name + "_silo");
        io::silo::save_mesh(load_mesh, out_name + "_silo");

        remove_path_if_exists(out_name + "_overlink");
        write_opts["file_style"] = "overlink";
        io::silo::save_mesh(load_mesh, out_name + "_overlink", write_opts);
    }
}

// TODO add tests for...
//  - materials once they are supported
//  - polytopal meshes once they are supported
//  - units once they are supported
//  - etc.
