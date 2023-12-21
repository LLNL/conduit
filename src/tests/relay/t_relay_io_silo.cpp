// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_io_silo.cpp
///
//-----------------------------------------------------------------------------

#include "silo_test_utils.hpp"

#include "conduit_relay.hpp"
#include "conduit_relay_io_silo.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
using namespace conduit::relay;

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

    n.print();

    io::silo_write(n,"tout_cold_storage_test.silo:myobj");

    Node n_load;
    io::silo_read("tout_cold_storage_test.silo:myobj",n_load);

    std::cout << "round trip" << std::endl;
    n_load.print();

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
        std::string path = utils::join_file_path("overlink", filename_vec.at(i));
        std::string input_file = relay_test_silo_data_path(path);
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
        if (mesh_type == "uniform")
        {
            silo_uniform_to_rect_conversion("coords", "mesh", save_mesh);
        }
        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
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

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
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
        }

        EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
        NodeConstIterator l_itr = load_mesh.children();
        NodeConstIterator s_itr = save_mesh.children();
        while (l_itr.has_next())
        {
            const Node &l_curr = l_itr.next();
            const Node &s_curr = s_itr.next();

            EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
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
    silo_name_changer("mesh", save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
}

//-----------------------------------------------------------------------------
// test material write and read
TEST(conduit_relay_io_silo, round_trip_venn)
{
    const std::vector<std::string> matset_types = {
        "full",
        "sparse_by_material",
        "sparse_by_element",
    };

    for (int i = 0; i < matset_types.size(); i ++)
    {
        std::string matset_type = matset_types[i];
        for (int j = 0; j < 2; j ++)
        {
            Node save_mesh, sbe, load_mesh, info;
            std::string size;
            int nx, ny;
            const double radius = 0.25;
            if (j == 0)
            {
                size = "small";
                nx = ny = 4;
            }
            else
            {
                size = "large";
                nx = ny = 100;
            }
            blueprint::mesh::examples::venn(matset_type, nx, ny, radius, save_mesh);
            // TODO remove this later when we support creating "full" and "sparse_by_material"
            // on read
            if (matset_type != "sparse_by_element")
            {
                // I create a second venn that is sparse by element b/c load_mesh will
                // convert silo data to sparse by element blueprint data. I need this to
                // diff successfully.
                blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, sbe);
            }

            const std::string basename = "silo_venn_" + matset_type + "_" + size;
            const std::string filename = basename + ".root";

            remove_path_if_exists(filename);
            io::silo::save_mesh(save_mesh, basename);
            io::silo::load_mesh(filename, load_mesh);
            EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

            // The silo reader creates sparse_by_element data
            if (matset_type != "sparse_by_element")
            {
                save_mesh.set_external(sbe);
            }

            // make changes to save mesh so the diff will pass

            // The field mat_check has values that are one type and matset_values
            // that are another type. The silo writer converts both to double arrays
            // in this case, so we follow suit.
            Node mat_check_new_values, mat_check_new_matset_values;
            save_mesh["fields"]["mat_check"]["values"].to_double_array(mat_check_new_values);
            save_mesh["fields"]["mat_check"]["matset_values"].to_double_array(mat_check_new_matset_values);
            save_mesh["fields"]["mat_check"]["values"].set_external(mat_check_new_values);
            save_mesh["fields"]["mat_check"]["matset_values"].set_external(mat_check_new_matset_values);

            silo_name_changer("mesh", save_mesh);

            // the loaded mesh will be in the multidomain format
            // but the saved mesh is in the single domain format
            EXPECT_EQ(load_mesh.number_of_children(), 1);
            EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
            EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_venn_modded_matnos)
{
    const std::string matset_type = "sparse_by_element";
    Node save_mesh, load_mesh, info;
    const int nx = 4;
    const int ny = 4;
    const double radius = 0.25;
    blueprint::mesh::examples::venn(matset_type, nx, ny, radius, save_mesh);

    auto replace_matno = [](int matno)
    {
        return (matno == 1 ? 15 : 
               (matno == 2 ? 37 : 
               (matno == 3 ? 4  : 
               (matno == 0 ? 22 : 
               -1))));
    };

    auto matmap_itr = save_mesh["matsets"]["matset"]["material_map"].children();
    while (matmap_itr.has_next())
    {
        Node &mat = matmap_itr.next();
        mat.set(replace_matno(mat.as_int()));
    }

    int_array matids = save_mesh["matsets"]["matset"]["material_ids"].value();
    for (int i = 0; i < save_mesh["matsets"]["matset"]["material_ids"].dtype().number_of_elements(); i ++)
    {
        matids[i] = replace_matno(matids[i]);
    }

    const std::string silo_basename = "silo_venn_" + matset_type + "_modded_matnos";
    const std::string silo_filename = silo_basename + ".root";
    remove_path_if_exists(silo_filename);
    io::silo::save_mesh(save_mesh, silo_basename);

    const std::string bp_basename = "bp_venn_" + matset_type + "_modded_matnos";
    const std::string bp_filename = bp_basename + ".root";
    remove_path_if_exists(bp_filename);
    io::blueprint::save_mesh(save_mesh, bp_basename, "hdf5");
    
    io::silo::load_mesh(silo_filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    // make changes to save mesh so the diff will pass

    // The field mat_check has values that are one type and matset_values
    // that are another type. The silo writer converts both to double arrays
    // in this case, so we follow suit.
    Node mat_check_new_values, mat_check_new_matset_values;
    save_mesh["fields"]["mat_check"]["values"].to_double_array(mat_check_new_values);
    save_mesh["fields"]["mat_check"]["matset_values"].to_double_array(mat_check_new_matset_values);
    save_mesh["fields"]["mat_check"]["values"].set_external(mat_check_new_values);
    save_mesh["fields"]["mat_check"]["matset_values"].set_external(mat_check_new_matset_values);

    // to_silo is going to reorder mixed materials least to greatest
    // so we must do the same
    int_array mat_ids = save_mesh["matsets"]["matset"]["material_ids"].value();
    const auto mat_id10 = mat_ids[10];
    const auto mat_id11 = mat_ids[11];
    const auto mat_id12 = mat_ids[12];
    mat_ids[10] = mat_id12;
    mat_ids[11] = mat_id10;
    mat_ids[12] = mat_id11;
    auto field_itr = save_mesh["fields"].children();
    while (field_itr.has_next())
    {
        const Node &n_field = field_itr.next();
        if (n_field.has_child("matset"))
        {
            double_array matset_vals = n_field["matset_values"].value();
            const auto matset_val10 = matset_vals[10];
            const auto matset_val11 = matset_vals[11];
            const auto matset_val12 = matset_vals[12];
            matset_vals[10] = matset_val12;
            matset_vals[11] = matset_val10;
            matset_vals[12] = matset_val11;
        }
    }

    silo_name_changer("mesh", save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_spiral_multi_dom_materials)
{
    Node save_mesh, load_mesh, info;
    const int ndomains = 4;
    blueprint::mesh::examples::spiral(ndomains, save_mesh);
    add_matset_to_spiral(save_mesh, ndomains);
    EXPECT_TRUE(blueprint::mesh::verify(save_mesh, info));
    
    const std::string basename = "silo_multidom_materials_spiral";
    const std::string filename = basename + ".cycle_000000.root";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename);
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

    // make changes to save mesh so the diff will pass
    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        // get the matset for this domain
        Node &n_matset = save_mesh[child]["matsets"]["matset"];

        // clean up volume fractions
        Node vf_arr;
        n_matset["volume_fractions"].to_float64_array(vf_arr);
        n_matset["volume_fractions"].reset();
        n_matset["volume_fractions"].set(vf_arr);
        
        // cheat a little bit - we don't have these to start
        n_matset["sizes"].set_external(load_mesh[child]["matsets"]["mesh_matset"]["sizes"]);
        n_matset["offsets"].set_external(load_mesh[child]["matsets"]["mesh_matset"]["offsets"]);

        silo_name_changer("mesh", save_mesh[child]);
    }

    EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
    NodeConstIterator l_itr = load_mesh.children();
    NodeConstIterator s_itr = save_mesh.children();
    while (l_itr.has_next())
    {
        const Node &l_curr = l_itr.next();
        const Node &s_curr = s_itr.next();

        EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_grid_adjset)
{
    Node save_mesh, load_mesh, info;
    blueprint::mesh::examples::grid("structured", 3, 3, 1, 2, 2, 1, save_mesh);

    const std::string basename = "silo_grid_adjset";
    const std::string filename = basename + "/OvlTop.silo";

    Node opts;
    opts["file_style"] = "overlink";
    opts["ovl_topo_name"] = "mesh";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename, opts);
    io::blueprint::save_mesh(save_mesh, basename, "hdf5");
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        // separate out vector fields
        Node &field_vel = save_mesh[child]["fields"]["vel"];
        Node &field_vel_u = save_mesh[child]["fields"]["vel_u"];
        Node &field_vel_v = save_mesh[child]["fields"]["vel_v"];
        
        field_vel_u["topology"].set(field_vel["topology"]);
        field_vel_u["association"].set(field_vel["association"]);
        field_vel_u["values"].set(field_vel["values/u"]);
        field_vel_v["topology"].set(field_vel["topology"]);
        field_vel_v["association"].set(field_vel["association"]);
        field_vel_v["values"].set(field_vel["values/v"]);

        save_mesh[child]["fields"].remove_child("vel");

        // make adjset pairwise
        Node &pairwise_adjset = save_mesh[child]["adjsets"]["adjset"];
        conduit::blueprint::mesh::adjset::to_pairwise(save_mesh[child]["adjsets"]["mesh_adj"], pairwise_adjset);
        save_mesh[child]["adjsets"].remove_child("mesh_adj");

        // make changes to save mesh so the diff will pass
        overlink_name_changer(save_mesh[child]);
    }

    EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
    NodeConstIterator l_itr = load_mesh.children();
    NodeConstIterator s_itr = save_mesh.children();
    while (l_itr.has_next())
    {
        const Node &l_curr = l_itr.next();
        const Node &s_curr = s_itr.next();

        EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
    }
}

//-----------------------------------------------------------------------------
// 
// test read and write semantics
// 

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, read_and_write_semantics)
{
    for (int ndomains = 2; ndomains < 6; ndomains ++)
    {
        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::spiral(ndomains, save_mesh);

        const std::string basename = "silo_spiral_" + std::to_string(ndomains) + "_domains";
        const std::string filename = basename + ".cycle_000000.root";

        remove_path_if_exists(filename);
        io::silo::write_mesh(save_mesh, basename);
        io::silo::read_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

        // make changes to save mesh so the diff will pass
        for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
        {
            silo_name_changer("mesh", save_mesh[child]);
        }

        EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
        NodeConstIterator l_itr = load_mesh.children();
        NodeConstIterator s_itr = save_mesh.children();
        while (l_itr.has_next())
        {
            const Node &l_curr = l_itr.next();
            const Node &s_curr = s_itr.next();

            EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
        }
    }
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
    }
    save_mesh[2].remove_child("fields");

    EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
    NodeConstIterator l_itr = load_mesh.children();
    NodeConstIterator s_itr = save_mesh.children();
    while (l_itr.has_next())
    {
        const Node &l_curr = l_itr.next();
        const Node &s_curr = s_itr.next();

        EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
    }
}

//-----------------------------------------------------------------------------
// matset is not defined on a domain
// 
// tests the silo "EMPTY" capability
TEST(conduit_relay_io_silo, missing_domain_matset)
{
    Node save_mesh, load_mesh, info;
    const int ndomains = 4;
    blueprint::mesh::examples::spiral(ndomains, save_mesh);
    add_matset_to_spiral(save_mesh, ndomains);
    EXPECT_TRUE(blueprint::mesh::verify(save_mesh, info));

    // remove information for a particular domain
    save_mesh[2]["matsets"].remove_child("matset");

    const std::string basename = "silo_missing_domain_matset_spiral";
    const std::string filename = basename + ".cycle_000000.root";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename);
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

    // make changes to save mesh so the diff will pass
    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        if (save_mesh[child].has_path("matsets/matset"))
        {
            // get the matset for this domain
            Node &n_matset = save_mesh[child]["matsets"]["matset"];

            // clean up volume fractions
            Node vf_arr;
            n_matset["volume_fractions"].to_float64_array(vf_arr);
            n_matset["volume_fractions"].reset();
            n_matset["volume_fractions"].set(vf_arr);
            
            // cheat a little bit - we don't have these to start
            n_matset["sizes"].set_external(load_mesh[child]["matsets"]["mesh_matset"]["sizes"]);
            n_matset["offsets"].set_external(load_mesh[child]["matsets"]["mesh_matset"]["offsets"]);
        }

        silo_name_changer("mesh", save_mesh[child]);
    }
    save_mesh[2].remove_child("matsets");

    EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
    NodeConstIterator l_itr = load_mesh.children();
    NodeConstIterator s_itr = save_mesh.children();
    while (l_itr.has_next())
    {
        const Node &l_curr = l_itr.next();
        const Node &s_curr = s_itr.next();

        EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
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
    }

    EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
    NodeConstIterator l_itr = load_mesh.children();
    NodeConstIterator s_itr = save_mesh.children();
    while (l_itr.has_next())
    {
        const Node &l_curr = l_itr.next();
        const Node &s_curr = s_itr.next();

        EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
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
    }

    // we must merge the two meshes in load mesh
    // the indexing is tricky because one is missing a domain
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

        EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
    }
}

//-----------------------------------------------------------------------------
// explicit points (unstructured mesh) do not use every coord
TEST(conduit_relay_io_silo, unstructured_points)
{
    Node save_mesh, load_mesh, info;
    blueprint::mesh::examples::braid("points", 2, 2, 2, save_mesh);

    std::vector<int> new_conn;
    std::vector<float> new_field1;
    std::vector<float> new_field2;
    std::vector<float64> new_xcoords, new_ycoords, new_zcoords;

    int_accessor conn = save_mesh["topologies"]["mesh"]["elements"]["connectivity"].value();

    float_accessor field1 = save_mesh["fields"]["braid"]["values"].value();
    float_accessor field2 = save_mesh["fields"]["radial"]["values"].value();

    float_accessor xcoords = save_mesh["coordsets"]["coords"]["values"]["x"].value();
    float_accessor ycoords = save_mesh["coordsets"]["coords"]["values"]["y"].value();
    float_accessor zcoords = save_mesh["coordsets"]["coords"]["values"]["z"].value();

    for (int i = 1; i < conn.number_of_elements(); i += 2)
    {
        new_conn.push_back(conn[i]);
        new_field1.push_back(field1[i]);
        new_field2.push_back(field2[i]);

        new_xcoords.push_back(xcoords[conn[i]]);
        new_ycoords.push_back(ycoords[conn[i]]);
        new_zcoords.push_back(zcoords[conn[i]]);
    }
    save_mesh["topologies"]["mesh"]["elements"]["connectivity"].reset();
    save_mesh["topologies"]["mesh"]["elements"]["connectivity"].set(new_conn);

    save_mesh["fields"].remove_child("vel");
    save_mesh["fields"]["braid"]["values"].reset();
    save_mesh["fields"]["braid"]["values"].set(new_field1);
    save_mesh["fields"]["radial"]["values"].reset();
    save_mesh["fields"]["radial"]["values"].set(new_field2);

    // we have modified braid such that it only uses half of the points in the coordset

    const std::string basename = "silo_unstructured_points_braid";
    const std::string filename = basename + ".cycle_000100.root";

    // remove existing root file, directory and any output files
    remove_path_if_exists(filename);

    io::silo::save_mesh(save_mesh, basename);
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    // now we must remove the unused points and change to an implicit points topo so that the diff passes
    save_mesh["coordsets"]["coords"]["values"]["x"].reset();
    save_mesh["coordsets"]["coords"]["values"]["x"].set(new_xcoords);
    save_mesh["coordsets"]["coords"]["values"]["y"].reset();
    save_mesh["coordsets"]["coords"]["values"]["y"].set(new_ycoords);
    save_mesh["coordsets"]["coords"]["values"]["z"].reset();
    save_mesh["coordsets"]["coords"]["values"]["z"].set(new_zcoords);

    save_mesh["topologies"].remove_child("mesh");
    save_mesh["topologies"]["mesh"]["type"] = "points";
    save_mesh["topologies"]["mesh"]["coordset"] = "coords";

    // the association doesn't matter for point meshes
    // we choose vertex by convention
    save_mesh["fields"]["radial"]["association"].reset();
    save_mesh["fields"]["radial"]["association"] = "vertex";

    silo_name_changer("mesh", save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
}

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
///      root_file_ext: "default", "root", "silo"
///            "default"   ==> "root"
///            if overlink, this parameter is unused.
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
            }

            EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
            NodeConstIterator l_itr = load_mesh.children();
            NodeConstIterator s_itr = save_mesh.children();
            while (l_itr.has_next())
            {
                const Node &l_curr = l_itr.next();
                const Node &s_curr = s_itr.next();

                EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_number_of_files)
{
    const std::vector<int> number_of_files = {-1, 2};
    for (int i = 0; i < number_of_files.size(); i ++)
    {
        Node opts;
        opts["file_style"] = "multi_file";
        opts["number_of_files"] = number_of_files[i];

        const std::string basename = "silo_save_option_number_of_files_" + 
                                     std::to_string(number_of_files[i]) + 
                                     "_spiral";
        const std::string filename = basename + ".cycle_000000.root";

        const int ndomains = 5;

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
        }

        EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
        NodeConstIterator l_itr = load_mesh.children();
        NodeConstIterator s_itr = save_mesh.children();
        while (l_itr.has_next())
        {
            const Node &l_curr = l_itr.next();
            const Node &s_curr = s_itr.next();

            EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
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
            save_mesh["state/cycle"] = 5;
        }

        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename, opts);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_root_file_ext)
{
    const std::vector<std::string> root_file_exts = {"default", "root", "silo"};

    for (int i = 0; i < root_file_exts.size(); i ++)
    {
        Node opts;
        opts["root_file_ext"] = root_file_exts[i];

        std::string actual_file_ext = root_file_exts[i];
        if (actual_file_ext == "default")
        {
            actual_file_ext = "root";
        }

        const std::string basename = "round_trip_save_option_root_file_ext_" + 
                                     root_file_exts[i] + "_basic";
        const std::string filename = basename + "." + actual_file_ext;

        Node save_mesh, load_mesh, info;
        blueprint::mesh::examples::basic("rectilinear", 3, 4, 0, save_mesh);
        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename, opts);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
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

    silo_name_changer("mymesh", save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_read_option_mesh_name)
{
    Node load_mesh, info, opts;
    const std::string path = utils::join_file_path("silo", "multi_curv3d.silo");
    const std::string input_file = relay_test_silo_data_path(path);

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
        
        silo_name_changer("mesh", save_mesh);

        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_io_silo, round_trip_save_option_overlink1)
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
        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename, opts);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

        for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
        {
            overlink_name_changer(save_mesh[child]);
        }

        EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
        NodeConstIterator l_itr = load_mesh.children();
        NodeConstIterator s_itr = save_mesh.children();
        while (l_itr.has_next())
        {
            const Node &l_curr = l_itr.next();
            const Node &s_curr = s_itr.next();

            EXPECT_FALSE(l_curr.diff(s_curr, info, CONDUIT_EPSILON, true));
        }
    }
}

//-----------------------------------------------------------------------------
// this tests var attributes and padding dimensions
TEST(conduit_relay_io_silo, round_trip_save_option_overlink2)
{
    const std::string basename = "silo_save_option_overlink_basic";
    const std::string filename = basename + "/OvlTop.silo";

    Node opts;
    opts["file_style"] = "overlink";

    Node save_mesh, load_mesh, info;
    blueprint::mesh::examples::basic("structured", 3, 3, 1, save_mesh);

    // add another field that is volume dependent
    Node &field2 = save_mesh["fields"]["field2"];
    field2["association"] = "element";
    field2["topology"] = "mesh";
    field2["volume_dependent"] = "true";
    field2["values"].set_external(save_mesh["fields"]["field"]["values"]);

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename, opts);
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

    // make changes to save mesh so the diff will pass
    overlink_name_changer(save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));

    // open silo files and do some checks

    DBfile *rootfile = DBOpen(filename.c_str(), DB_UNKNOWN, DB_READ);
    EXPECT_TRUE(DBInqVarExists(rootfile, "VAR_ATTRIBUTES"));
    EXPECT_TRUE(DBInqVarType(rootfile, "VAR_ATTRIBUTES") == DB_ARRAY);

    DBcompoundarray *var_attr = DBGetCompoundarray(rootfile, "VAR_ATTRIBUTES");

    // fetch pointers to elements inside the compound array
    char **elemnames = var_attr->elemnames;
    int *elemlengths = var_attr->elemlengths;
    int nelems       = var_attr->nelems;
    int *values      = static_cast<int *>(var_attr->values);
    int nvalues      = var_attr->nvalues;
    int datatype     = var_attr->datatype;

    EXPECT_EQ(std::string(elemnames[0]), "field");
    EXPECT_EQ(std::string(elemnames[1]), "field2");
    EXPECT_EQ(elemlengths[0], 5);
    EXPECT_EQ(elemlengths[1], 5);
    EXPECT_EQ(nelems, 2);
    // for first var
    EXPECT_EQ(values[0], 1);
    EXPECT_EQ(values[1], 0);
    EXPECT_EQ(values[2], 1);
    EXPECT_EQ(values[3], 0);
    EXPECT_EQ(values[4], 1);
    // for second var
    EXPECT_EQ(values[5], 1);
    EXPECT_EQ(values[6], 1);
    EXPECT_EQ(values[7], 1);
    EXPECT_EQ(values[8], 0);
    EXPECT_EQ(values[9], 1);
    EXPECT_EQ(nvalues, 10);
    EXPECT_EQ(datatype, DB_INT);

    DBFreeCompoundarray(var_attr);

    EXPECT_TRUE(DBInqVarExists(rootfile, "PAD_DIMS"));
    EXPECT_TRUE(DBInqVarType(rootfile, "PAD_DIMS") == DB_ARRAY);

    DBcompoundarray *pad_dims = DBGetCompoundarray(rootfile, "PAD_DIMS");

    // fetch pointers to elements inside the compound array
    elemnames   = pad_dims->elemnames;
    elemlengths = pad_dims->elemlengths;
    nelems      = pad_dims->nelems;
    values      = static_cast<int *>(pad_dims->values);
    nvalues     = pad_dims->nvalues;
    datatype    = pad_dims->datatype;

    EXPECT_EQ(std::string(elemnames[0]), "paddims");
    EXPECT_EQ(elemlengths[0], 6);
    EXPECT_EQ(nelems, 1);
    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[1], 0);
    EXPECT_EQ(values[2], 0);
    EXPECT_EQ(values[3], 0);
    EXPECT_EQ(values[4], 0);
    EXPECT_EQ(values[5], 0);
    EXPECT_EQ(nvalues, 6);
    EXPECT_EQ(datatype, DB_INT);

    DBFreeCompoundarray(pad_dims);

    DBClose(rootfile);

    // now check domain file

    const std::string dom_filename = basename + "/domain0.silo";
    DBfile *domfile = DBOpen(dom_filename.c_str(), DB_UNKNOWN, DB_READ);

    EXPECT_TRUE(DBInqVarExists(domfile, "DOMAIN_NEIGHBOR_NUMS"));
    EXPECT_TRUE(DBInqVarType(domfile, "DOMAIN_NEIGHBOR_NUMS") == DB_ARRAY);

    DBcompoundarray *dom_neighbor_nums = DBGetCompoundarray(domfile, "DOMAIN_NEIGHBOR_NUMS");

    // fetch pointers to elements inside the compound array
    elemnames   = dom_neighbor_nums->elemnames;
    elemlengths = dom_neighbor_nums->elemlengths;
    nelems      = dom_neighbor_nums->nelems;
    values      = static_cast<int *>(dom_neighbor_nums->values);
    nvalues     = dom_neighbor_nums->nvalues;
    datatype    = dom_neighbor_nums->datatype;

    EXPECT_EQ(std::string(elemnames[0]), "num_neighbors");
    EXPECT_EQ(std::string(elemnames[1]), "neighbor_nums");
    EXPECT_EQ(elemlengths[0], 1);
    EXPECT_EQ(elemlengths[1], 0);
    EXPECT_EQ(nelems, 2);
    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(nvalues, 1);
    EXPECT_EQ(datatype, DB_INT);

    DBFreeCompoundarray(dom_neighbor_nums);

    DBClose(domfile);
}

//-----------------------------------------------------------------------------
// this tests material i/o
TEST(conduit_relay_io_silo, round_trip_save_option_overlink3)
{
    Node save_mesh, load_mesh, info;
    const int nx = 100, ny = 100;
    const double radius = 0.25;
    blueprint::mesh::examples::venn("sparse_by_element", nx, ny, radius, save_mesh);

    const std::string basename = "silo_save_option_overlink_venn";
    const std::string filename = basename + "/OvlTop.silo";

    Node opts;
    opts["file_style"] = "overlink";

    remove_path_if_exists(filename);
    io::silo::save_mesh(save_mesh, basename, opts);
    io::silo::load_mesh(filename, load_mesh);
    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    // make changes to save mesh so the diff will pass

    // The field mat_check has values that are one type and matset_values
    // that are another type. The silo writer converts both to double arrays
    // in this case, so we follow suit.
    Node mat_check_new_values, mat_check_new_matset_values;
    save_mesh["fields"]["mat_check"]["values"].to_double_array(mat_check_new_values);
    save_mesh["fields"]["mat_check"]["matset_values"].to_double_array(mat_check_new_matset_values);
    save_mesh["fields"]["mat_check"]["values"].set_external(mat_check_new_values);
    save_mesh["fields"]["mat_check"]["matset_values"].set_external(mat_check_new_matset_values);

    overlink_name_changer(save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

    EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
}

//-----------------------------------------------------------------------------
// we are testing vector fields get converted to scalars
TEST(conduit_relay_io_silo, round_trip_save_option_overlink4)
{
    const std::vector<std::pair<std::string, std::string>> mesh_types = {
        std::make_pair("rectilinear", "2"), std::make_pair("rectilinear", "3"),
        std::make_pair("structured", "2"), std::make_pair("structured", "3"),
        std::make_pair("quads", "2"),
        std::make_pair("hexs", "3"),
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

        const std::string basename = "silo_save_option_overlink_braid_" + mesh_type + "_" + dim + "D";
        const std::string filename = basename + "/OvlTop.silo";

        Node opts;
        opts["file_style"] = "overlink";

        // remove existing root file, directory and any output files
        remove_path_if_exists(filename);
        io::silo::save_mesh(save_mesh, basename, opts);
        io::silo::load_mesh(filename, load_mesh);
        EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

        Node &field_vel = save_mesh["fields"]["vel"];
        Node &field_vel_u = save_mesh["fields"]["vel_u"];
        Node &field_vel_v = save_mesh["fields"]["vel_v"];

        field_vel_u["topology"].set(field_vel["topology"]);
        field_vel_u["association"].set(field_vel["association"]);
        field_vel_u["values"].set(field_vel["values/u"]);
        field_vel_v["topology"].set(field_vel["topology"]);
        field_vel_v["association"].set(field_vel["association"]);
        field_vel_v["values"].set(field_vel["values/v"]);

        if (dim == "3")
        {
            Node &field_vel_w = save_mesh["fields"]["vel_w"];
            field_vel_w["topology"].set(field_vel["topology"]);
            field_vel_w["association"].set(field_vel["association"]);
            field_vel_w["values"].set(field_vel["values/w"]);
        }

        save_mesh["fields"].remove_child("vel");

        // make changes to save mesh so the diff will pass
        overlink_name_changer(save_mesh);
        
        // the loaded mesh will be in the multidomain format
        // but the saved mesh is in the single domain format
        EXPECT_EQ(load_mesh.number_of_children(), 1);
        EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

        EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
    }
}

//-----------------------------------------------------------------------------
// we are testing spec sets
TEST(conduit_relay_io_silo, round_trip_save_option_overlink5)
{
    Node save_mesh, load_mesh, info;
    // TODO do 10 x 10
    blueprint::mesh::examples::misc("specsets", 3, 3, 1, save_mesh);

    std::cout << save_mesh.to_yaml() << std::endl;

    const std::string basename = "silo_save_option_overlink_specsets";
    const std::string filename = basename + "/OvlTop.silo";

    Node opts;
    opts["file_style"] = "overlink";

    // // remove existing root file, directory and any output files
    // remove_path_if_exists(filename);
    // io::silo::save_mesh(save_mesh, basename, opts);
    // io::silo::load_mesh(filename, load_mesh);
    // EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    // // make changes to save mesh so the diff will pass
    // overlink_name_changer(save_mesh);
    
    // // the loaded mesh will be in the multidomain format
    // // but the saved mesh is in the single domain format
    // EXPECT_EQ(load_mesh.number_of_children(), 1);
    // EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

    // EXPECT_FALSE(load_mesh[0].diff(save_mesh, info, CONDUIT_EPSILON, true));
}

// //-----------------------------------------------------------------------------

// //
// // read and write Silo and Overlink tests
// //

// //-----------------------------------------------------------------------------
// // read normal silo files containing multimeshes, multivars, and multimats
// TEST(conduit_relay_io_silo, read_silo)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//         {".",                  "multi_curv3d", ".silo", ""            }, // test default case
//         {".",                  "multi_curv3d", ".silo", "mesh1"       },
//         {".",                  "multi_curv3d", ".silo", "mesh1_back"  },
//         {".",                  "multi_curv3d", ".silo", "mesh1_dup"   },
//         {".",                  "multi_curv3d", ".silo", "mesh1_front" },
//         {".",                  "multi_curv3d", ".silo", "mesh1_hidden"},
//         {".",                  "tire",         ".silo", ""            }, // test default case
//         {".",                  "tire",         ".silo", "tire"        },
//         {".",                  "galaxy0000",   ".silo", ""            }, // test default case
//         {".",                  "galaxy0000",   ".silo", "StarMesh"    },
//         {".",                  "emptydomains", ".silo", ""            }, // test default case
//         {".",                  "emptydomains", ".silo", "mesh"        },
//         {"multidir_test_data", "multidir0000", ".root", ""            }, // test default case
//         {"multidir_test_data", "multidir0000", ".root", "Mesh"        },
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;
//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("silo", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_silo_" + basename;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         // TODO are these remove paths doing anything? Don't they need filenames?
//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = meshname;
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// //-----------------------------------------------------------------------------
// // test that we can read the fake overlink files from the visit test data
// TEST(conduit_relay_io_silo, read_fake_overlink)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//      // {"ev_0_0_100",              "OvlTop", ".silo", ""     }, // test default case
//      // {"ev_0_0_100",              "OvlTop", ".silo", "MMESH"},
//         // uncomment once silo ucdmesh phzones are supported
//         {"hl18spec",                "OvlTop", ".silo", ""     }, // test default case
//         {"hl18spec",                "OvlTop", ".silo", "MMESH"},
//      // {"regrovl_qh_1000_10001_4", "OvlTop", ".silo", ""     }, // test default case
//      // {"regrovl_qh_1000_10001_4", "OvlTop", ".silo", "MMESH"},
//         // uncomment once silo ucdmesh phzones are supported
//         {"utpyr4",                  "OvlTop", ".silo", ""     }, // test default case
//         {"utpyr4",                  "OvlTop", ".silo", "MMESH"},
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;
//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("fake_overlink", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_fake_overlink_" + dirname;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = "MMESH";
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// //-----------------------------------------------------------------------------
// // read overlink files in symlink format
// // should be similar to reading raw silo
// TEST(conduit_relay_io_silo, read_overlink_symlink_format)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//         {".", "box2d",                  ".silo", ""     }, // test default case
//         {".", "box2d",                  ".silo", "MMESH"},
//         {".", "box3d",                  ".silo", ""     }, // test default case
//         {".", "box3d",                  ".silo", "MMESH"},
//      // {".", "diamond",                ".silo", ""     }, // test default case
//      // {".", "diamond",                ".silo", "MMESH"},
//         // fails b/c polytopal not yet supported
//         {".", "testDisk2D_a",           ".silo", ""     }, // test default case
//         {".", "testDisk2D_a",           ".silo", "MMESH"},
//      // {".", "donordiv.s2_materials2", ".silo", ""     }, // test default case
//      // {".", "donordiv.s2_materials2", ".silo", "MMESH"},
//         // fails b/c polytopal not yet supported
//         {".", "donordiv.s2_materials3", ".silo", ""     }, // test default case
//         {".", "donordiv.s2_materials3", ".silo", "MMESH"},
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;
//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("overlink", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_overlink_symlink_" + basename;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = "MMESH";
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// //-----------------------------------------------------------------------------
// // read overlink directly from ovltop.silo
// // this case is tricky and involves messing with paths
// TEST(conduit_relay_io_silo, read_overlink_directly)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//         {"box2d",                  "OvlTop", ".silo", ""     }, // test default case
//         {"box2d",                  "OvlTop", ".silo", "MMESH"},
//         {"box3d",                  "OvlTop", ".silo", ""     }, // test default case
//         {"box3d",                  "OvlTop", ".silo", "MMESH"},
//      // {"diamond",                "OvlTop", ".silo", ""     }, // test default case
//      // {"diamond",                "OvlTop", ".silo", "MMESH"},
//         {"testDisk2D_a",           "OvlTop", ".silo", ""     }, // test default case
//         {"testDisk2D_a",           "OvlTop", ".silo", "MMESH"},
//      // {"donordiv.s2_materials2", "OvlTop", ".silo", ""     }, // test default case
//      // {"donordiv.s2_materials2", "OvlTop", ".silo", "MMESH"},
//         {"donordiv.s2_materials3", "OvlTop", ".silo", ""     }, // test default case
//         {"donordiv.s2_materials3", "OvlTop", ".silo", "MMESH"},
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;

//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("overlink", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_overlink_direct_" + dirname;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = "MMESH";
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// // TODO add tests for...
// //  - polytopal meshes once they are supported
// //  - units once they are supported
// //  - etc.

// // TODO add tetra8 and c36_m5 to all the overlink i/o tests

// // TODO somewhere I need to error on overlink when there are different var or mesh types across domains
