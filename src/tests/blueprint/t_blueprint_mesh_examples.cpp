// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <math.h>
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

index_t OUTPUT_NUM_AXIS_POINTS = 5;

std::string PROTOCOL_VER = CONDUIT_VERSION;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_2d)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);

    bool silo_enabled = io_protos["io/protocols/conduit_silo"].as_string() == "enabled";
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";

    // we are using one node to hold group of example meshes purely out of convenience
    Node dsets;
    // can be overridden via command line
    index_t npts_x = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_y = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_z = 1; // 2D examples ...

    blueprint::mesh::examples::braid("uniform",
                                      npts_x,
                                      npts_y,
                                      npts_z,
                                      dsets["uniform"]);

    blueprint::mesh::examples::braid("rectilinear",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["rect"]);

    blueprint::mesh::examples::braid("structured",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["struct"]);

    blueprint::mesh::examples::braid("lines",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["lines"]);

    blueprint::mesh::examples::braid("tris",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["tris"]);

    blueprint::mesh::examples::braid("quads",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["quads"]);

    blueprint::mesh::examples::braid("quads_and_tris",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["quads_and_tris"]);

    blueprint::mesh::examples::braid("quads_and_tris_offsets",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["quads_and_tris_offsets"]);

    blueprint::mesh::examples::braid("points",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points"]);

    blueprint::mesh::examples::braid("points_implicit",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points_implicit"]);

    Node info;
    NodeConstIterator itr = dsets.children();
    while(itr.has_next())
    {
        const Node &mesh = itr.next();
        EXPECT_TRUE(blueprint::mesh::verify(mesh,info));
        CONDUIT_INFO(info.to_yaml());
    }

    // TODO: Add VisIt support for rendering mixed element and implicit point
    // meshes so they don't have to be removed before outputting mesh data.
    dsets.remove("quads_and_tris");
    dsets.remove("quads_and_tris_offsets");
    dsets.remove("points_implicit");

    relay::io_blueprint::save(dsets, "braid_2d_examples.blueprint_root");
    if(hdf5_enabled)
    {
        relay::io_blueprint::save(dsets, "braid_2d_examples.blueprint_root_hdf5");
    }
    if(silo_enabled)
    {
        // we removed datasets above, so we need an updated iterator
        itr = dsets.children();
        while(itr.has_next())
        {
            const Node &mesh = itr.next();
            std::string name = itr.name();

            // Skip output of silo mesh for mixed mesh of tris and quads for now.
            // The silo output is not yet defined and it throws an exception
            // in conduit_silo.cpp::silo_write_ucd_zonelist()
            // in the following line that is looking for the 'shape' node:
            //   std::string topo_shape = shape_block->fetch("shape").as_string();
            // which does not exist for indexed_stream meshes.
            // The silo writer needs to be updated for this case.
            if( name == "quads_and_tris" || name == "quads_and_tris_offsets" )
            {
                CONDUIT_INFO("\tNOTE: skipping output to SILO -- ")
                CONDUIT_INFO("feature is unavailable for mixed element meshes")
                continue;
            }

            relay::io::save(mesh,
                            "braid_2d_" + name +  "_example.silo:mesh",
                            "conduit_silo_mesh");
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_3d)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);

    bool silo_enabled = io_protos["io/protocols/conduit_silo"].as_string() == "enabled";
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";

    // we are using one node to hold group of example meshes purely out of convenience
    Node dsets;
    // can be overridden via command line
    index_t npts_x = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_y = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_z = OUTPUT_NUM_AXIS_POINTS; // 3D examples ...

    blueprint::mesh::examples::braid("uniform",
                                      npts_x,
                                      npts_y,
                                      npts_z,
                                      dsets["uniform"]);

    blueprint::mesh::examples::braid("rectilinear",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["rect"]);

    blueprint::mesh::examples::braid("structured",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["struct"]);

    blueprint::mesh::examples::braid("points",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points"]);

    blueprint::mesh::examples::braid("points_implicit",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points_implicit"]);

    blueprint::mesh::examples::braid("lines",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["lines"]);

    blueprint::mesh::examples::braid("tets",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["tets"]);

    blueprint::mesh::examples::braid("hexs",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["hexs"]);

    blueprint::mesh::examples::braid("hexs_and_tets",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["hexs_and_tets"]);

    Node info;
    NodeConstIterator itr = dsets.children();
    while(itr.has_next())
    {
        const Node &mesh = itr.next();
        EXPECT_TRUE(blueprint::mesh::verify(mesh,info));
        CONDUIT_INFO(info.to_yaml());
    }

    // TODO: Add VisIt support for rendering mixed element and implicit point
    // meshes so they don't have to be removed before outputting mesh data.
    dsets.remove("hexs_and_tets");
    dsets.remove("points_implicit");

    relay::io_blueprint::save(dsets, "braid_3d_examples.blueprint_root");
    if(hdf5_enabled)
    {
        relay::io_blueprint::save(dsets, "braid_3d_examples.blueprint_root_hdf5");
    }
    if(silo_enabled)
    {
        // we removed datasets above, so we need an updated iterator
        itr = dsets.children();
        while(itr.has_next())
        {
            const Node &mesh = itr.next();
            std::string name = itr.name();

            // Skip output of silo mesh for mixed mesh of hexs and tets for now.
            // The silo output is not yet defined and it throws an exception
            // in conduit_silo.cpp::silo_write_ucd_zonelist()
            // in the following line that is looking for the 'shape' node:
            //              std::string topo_shape = shape_block->fetch("shape").as_string();
            // which does not exist for indexed_stream meshes.
            // The silo writer needs to be updated for this case.
            if(name == "hexs_and_tets")
            {
                CONDUIT_INFO("\tNOTE: skipping output to SILO -- ")
                CONDUIT_INFO("feature is unavailable for mixed element meshes")
                continue;
            }

            relay::io::save(mesh,
                            "braid_3d_" + name +  "_example.silo:mesh",
                            "conduit_silo_mesh");
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, braid_too_small_npts)
{

    std::vector<std::string> braid_type_strings;


    braid_type_strings.push_back("uniform");
    braid_type_strings.push_back("lines");
    braid_type_strings.push_back("rectilinear");
    braid_type_strings.push_back("structured");

    // 2d extra
    braid_type_strings.push_back("tris");
    braid_type_strings.push_back("quads");


    // 3d extra

    braid_type_strings.push_back("tets");
    braid_type_strings.push_back("hexs");

    Node mesh;

    for(size_t i = 0; i < braid_type_strings.size(); i++)
    {
        mesh.reset();
        EXPECT_THROW(blueprint::mesh::examples::braid(braid_type_strings[i],
                                                      1,
                                                      1,
                                                      1,
                                                      mesh), conduit::Error);
    }

    braid_type_strings.clear();
    braid_type_strings.push_back("points");
    braid_type_strings.push_back("points_implicit");

    for(size_t i = 0; i < braid_type_strings.size(); i++)
    {
        mesh.reset();
        blueprint::mesh::examples::braid(braid_type_strings[i],
                                         1,
                                         1,
                                         1,
                                         mesh);
        Node info;
        EXPECT_TRUE(blueprint::mesh::verify(mesh,info));
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, julia)
{
    Node res;
    blueprint::mesh::examples::julia(500,   500, // nx, ny
                                     -2.0,  2.0, // x range
                                     -2.0,  2.0, // y range
                                     0.285, 0.01, // c value
                                     res);
    Node info;
    EXPECT_TRUE(blueprint::mesh::verify(res,info));
    CONDUIT_INFO(info.to_yaml());

    relay::io_blueprint::save(res, "julia_example.blueprint_root");
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, spiral)
{
    int ndoms = 10;
    Node res;
    blueprint::mesh::examples::spiral(ndoms,res["spiral"]);
    Node info;
    EXPECT_TRUE(blueprint::mesh::verify(res["spiral/domain_000000"],info));
    CONDUIT_INFO(info.to_yaml());

    blueprint::mesh::generate_index(res["spiral/domain_000000"],
                                    "",
                                    ndoms,
                                    res["blueprint_index/spiral"]);

    // save json
    res["protocol/name"] = "json";
    res["protocol/version"] = PROTOCOL_VER;

    res["number_of_files"] = 1;
    res["number_of_trees"] = ndoms;
    res["file_pattern"] = "spiral_example.blueprint_root";
    res["tree_pattern"] = "spiral/domain_%06d";

    CONDUIT_INFO("Creating: spiral_example.blueprint_root")
    relay::io::save(res,"spiral_example.blueprint_root","json");
}



//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, polytess)
{
    const index_t nlevels = 3;
    Node res;
    blueprint::mesh::examples::polytess(nlevels,
                                        res);

    Node info;
    EXPECT_TRUE(blueprint::mesh::verify(res,info));
    CONDUIT_INFO(info.to_yaml());

    relay::io_blueprint::save(res, "polytess_example.blueprint_root");
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, 2d_braid_zero_z_check)
{
    Node mesh, info;
    // these checks make sure braid generates valid fields even when
    // # of z pointers == 0
    int npts_x = 5;
    int npts_y = 5;
    int npts_z = 0;

    std::vector<std::string> braid_type_strings;
    braid_type_strings.push_back("points");
    braid_type_strings.push_back("points_implicit");
    braid_type_strings.push_back("lines");
    braid_type_strings.push_back("rectilinear");
    braid_type_strings.push_back("structured");
    braid_type_strings.push_back("tris");
    braid_type_strings.push_back("quads");

    for(size_t i = 0; i < braid_type_strings.size(); i++)
    {
        mesh.reset();
        blueprint::mesh::examples::braid(braid_type_strings[i],
                                          npts_x,
                                          npts_y,
                                          npts_z,
                                          mesh);
        // make the braid vertex-assoced field has with more than zero entries
        EXPECT_GT(mesh["fields/braid/values"].dtype().number_of_elements(),0);
        mesh.print();
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_misc)
{
    // for each "misc" mesh example, generate a small 2d mesh and
    // save to yaml
    std::vector<std::string> misc_type_strings;
    misc_type_strings.push_back("matsets");
    misc_type_strings.push_back("specsets");
    misc_type_strings.push_back("adjsets");
    misc_type_strings.push_back("nestsets");

    Node mesh;
    int npts_x = 5;
    int npts_y = 5;

    for(size_t i = 0; i < misc_type_strings.size(); i++)
    {
        mesh.reset();
        blueprint::mesh::examples::misc(misc_type_strings[i],
                                        npts_x,
                                        npts_y,
                                        1,
                                        mesh);
        mesh.save("misc_example_" + misc_type_strings[i] + ".yaml");
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, check_gen_index_state_prop)
{
    // braid provides cycle and time, make sure those make it into
    // an index created with generate_index
    Node mesh;
    blueprint::mesh::examples::braid("hexs",
                                      5,
                                      5,
                                      5,
                                      mesh);

    Node idx;
    blueprint::mesh::generate_index(mesh,
                                    "",
                                    1,
                                    idx);

    EXPECT_TRUE(idx.has_path("state/cycle"));
    EXPECT_TRUE(idx.has_path("state/time"));
}

//-----------------------------------------------------------------------------
void venn_test_small_yaml(const std::string &venn_type)
{
    // provide small example save to yaml for folks to look at
    const int nx = 25, ny = 25;
    const double radius = 0.25;

    Node res, info, n_idx;
    blueprint::mesh::examples::venn(venn_type, nx, ny, radius, res);
    blueprint::mesh::examples::venn(venn_type, nx, ny, radius, res);
    EXPECT_TRUE(blueprint::mesh::verify(res, info));
    blueprint::mesh::generate_index(res,"",1,n_idx);
    EXPECT_TRUE(blueprint::verify("mesh/index",n_idx,info));
    res.save("venn_small_example_" + venn_type + ".yaml");
}

//-----------------------------------------------------------------------------
void venn_test(const std::string &venn_type)
{
    const int nx = 100, ny = 100;
    const double radius = 0.25;

    Node res, info;
    blueprint::mesh::examples::venn(venn_type, nx, ny, radius, res);

    EXPECT_TRUE(blueprint::mesh::verify(res, info));
    utils::log::remove_valid(info);
    CONDUIT_INFO(info.to_yaml());
    CONDUIT_INFO(res.schema().to_json());

    std::string ofbase = "venn_example_" + venn_type;
    std::cout << "[Saving " << ofbase << "]" << std::endl;
    relay::io_blueprint::save(res, ofbase + ".blueprint_root");

    {
        std::cout << "[Verifying field area is correct]" << std::endl;

        Node &area_actual = res["fields/area/values"];
        Node area_expected;
        area_expected.set(std::vector<double>(
            area_actual.dtype().number_of_elements(), 1.0 / (nx * ny)));

        bool actual_matches_expected = !area_expected.diff(area_actual, info);
        EXPECT_TRUE(actual_matches_expected);
        if(!actual_matches_expected)
        {
            CONDUIT_INFO(info.to_yaml());
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, venn_full)
{
    venn_test("full");
    venn_test_small_yaml("full");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, venn_sparse_by_material)
{
    venn_test("sparse_by_material");
    venn_test_small_yaml("sparse_by_material");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, venn_sparse_by_element)
{
    venn_test("sparse_by_element");
    venn_test_small_yaml("sparse_by_element");
}


TEST(conduit_blueprint_mesh_examples, mesh_julia_nestset_simple)
{

    Node mesh;

    Node res;
    blueprint::mesh::examples::julia_nestsets_simple(-2.0,  2.0, // x range
                                                     -2.0,  2.0, // y range
                                                     0.285, 0.01, // c value
                                                     res["julia_nestset_simple"]);

    int ndoms = res["julia_nestset_simple"].number_of_children();

    Node info;
    if(!blueprint::mesh::verify(res["julia_nestset_simple"],info))
    {
      info.print();
    }

    blueprint::mesh::generate_index(res["julia_nestset_simple/domain_000000"],
                                    "",
                                    ndoms,
                                    res["blueprint_index/julia_nestset_simple"]);

    // save json
    res["protocol/name"] = "json";
    res["protocol/version"] = PROTOCOL_VER;

    res["number_of_files"] = 1;
    res["number_of_trees"] = ndoms;
    res["file_pattern"] = "julia_nestset_simple.blueprint_root";
    res["tree_pattern"] = "julia_nestset_simple/domain_%06d";

    CONDUIT_INFO("Creating: julia_nestset_simple.blueprint_root")
    relay::io::save(res,"julia_nestset_simple.blueprint_root","json");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_julia_nestset)
{

    Node mesh;

    Node res;
    blueprint::mesh::examples::julia_nestsets_complex(20,   20, // nx, ny
                                                      -2.0,  2.0, // x range
                                                      -2.0,  2.0, // y range
                                                      0.285, 0.01, // c value
                                                      3, // amr levels
                                                      res["julia_nestset_complex"]);

    int ndoms = res["julia_nestset_complex"].number_of_children();

    Node info;
    if(!blueprint::mesh::verify(res["julia_nestset_complex"],info))
    {
      info.print();
    }

    blueprint::mesh::generate_index(res["julia_nestset_complex/domain_000000"],
                                    "",
                                    ndoms,
                                    res["blueprint_index/julia_nestset_complex"]);

    // save json
    res["protocol/name"] = "json";
    res["protocol/version"] = PROTOCOL_VER;

    res["number_of_files"] = 1;
    res["number_of_trees"] = ndoms;
    res["file_pattern"] = "julia_nestset_complex_example.blueprint_root";
    res["tree_pattern"] = "julia_nestset_complex/domain_%06d";

    CONDUIT_INFO("Creating: julia_nestset_complex_example.blueprint_root")
    relay::io::save(res,"julia_nestset_complex_example.blueprint_root","json");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, save_adjset_uniform)
{

    Node io_protos;
    relay::io::about(io_protos["io"]);
    bool hdf5_enabled =io_protos["io/protocols/hdf5"].as_string() == "enabled";

    // skip if we don't have hdf5 since this example has several domains
    if(!hdf5_enabled)
        return;

    Node mesh,idx;
    blueprint::mesh::examples::adjset_uniform(mesh);
    blueprint::mesh::generate_index(mesh[0],
                                    "",
                                    8,
                                    mesh["blueprint_index/adj_uniform"]);

    mesh["protocol/name"] = "hdf5";
    mesh["protocol/version"] = PROTOCOL_VER;

    mesh["number_of_files"] = 1;
    mesh["number_of_trees"] = 8;
    mesh["file_pattern"] = "adj_uniform_example.blueprint_root";
    mesh["tree_pattern"] = "domain_%06d";

    CONDUIT_INFO("Creating: adj_uniform_example.blueprint_root")
    relay::io::save(mesh,"adj_uniform_example.blueprint_root","json");
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, basic_bad_inputs)
{
    Node res;

    // several with bad inputs
    EXPECT_THROW(blueprint::mesh::examples::basic("uniform",
                                                  -1,
                                                  2,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::basic("uniform",
                                                  1,
                                                  1,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::basic("uniform",
                                                  2,
                                                  -1,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::basic("tets",
                                                  2,
                                                  2,
                                                  1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::basic("hexs",
                                                  2,
                                                  2,
                                                  0,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::basic("polyhedra",
                                                  2,
                                                  2,
                                                  -1,
                                                  res),conduit::Error);

    // a few ok
    blueprint::mesh::examples::basic("uniform",
                                     2,
                                     2,
                                     2,
                                     res);

    blueprint::mesh::examples::basic("tets",
                                     2,
                                     2,
                                     2,
                                     res);

}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, braid_bad_inputs)
{
    Node res;

    // several with bad inputs
    EXPECT_THROW(blueprint::mesh::examples::braid("uniform",
                                                  -1,
                                                  2,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("uniform",
                                                  1,
                                                  1,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("uniform",
                                                  2,
                                                  -1,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("tets",
                                                  2,
                                                  2,
                                                  1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("hexs",
                                                  2,
                                                  2,
                                                  0,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("hexs_poly",
                                                  2,
                                                  2,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("points",
                                                  0,
                                                  0,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("points_implicit",
                                                  0,
                                                  0,
                                                  -1,
                                                  res),conduit::Error);

    EXPECT_THROW(blueprint::mesh::examples::braid("points",
                                                  1,
                                                  1,
                                                  -1,
                                                  res),conduit::Error);

    // a few ok
    blueprint::mesh::examples::braid("points",
                                     1,
                                     1,
                                     0,
                                     res);

    res.print();
    // check conn array, should have 1 entry
    EXPECT_EQ(res["topologies/mesh/elements/connectivity"].dtype().number_of_elements(),1);

    // should be 2d
    EXPECT_EQ(res["coordsets/coords/values"].number_of_children(),2);

    blueprint::mesh::examples::braid("points",
                                     1,
                                     1,
                                     2,
                                     res);
    // should be 3d
    EXPECT_EQ(res["coordsets/coords/values"].number_of_children(),3);


    blueprint::mesh::examples::braid("uniform",
                                     2,
                                     2,
                                     2,
                                     res);

    blueprint::mesh::examples::braid("tets",
                                     2,
                                     2,
                                     2,
                                     res);

}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, braid_diff_dims)
{
    Node mesh;
    conduit::blueprint::mesh::examples::braid("quads", 2, 3, 0, mesh);
    conduit::relay::io_blueprint::save(mesh, "braid_quads_2_3.blueprint_root");

    conduit::blueprint::mesh::examples::braid("tris", 2, 3, 0, mesh);
    conduit::relay::io_blueprint::save(mesh, "braid_tris_2_3.blueprint_root");

    conduit::blueprint::mesh::examples::braid("tets", 2, 3, 4, mesh);
    conduit::relay::io_blueprint::save(mesh, "braid_tets_2_3_4.blueprint_root");

    conduit::blueprint::mesh::examples::braid("hexs", 2, 3, 4, mesh);
    conduit::relay::io_blueprint::save(mesh, "braid_hexs_2_3_4.blueprint_root");

}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, number_of_domains)
{
    // test both multi dom and non multi dom case

    Node data;
    // braid
    blueprint::mesh::examples::braid("uniform",
                                      10,10,10,
                                      data);

    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(data),1);

    // now check a multi domain mesh ...

    // spiral with 3 domains
    conduit::blueprint::mesh::examples::spiral(3,data);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(data),3);

    // spiral with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(data),7);


}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    // allow override of the data size via the command line
    if(argc == 2)
    {
        OUTPUT_NUM_AXIS_POINTS = atoi(argv[1]);
    }

    result = RUN_ALL_TESTS();
    return result;
}
