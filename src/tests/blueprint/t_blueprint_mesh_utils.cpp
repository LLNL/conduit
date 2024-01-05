// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_utils.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include "blueprint_test_helpers.hpp"

#include <algorithm>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
using namespace generate;

//---------------------------------------------------------------------------
/**
 @brief Save the node to an HDF5 compatible with VisIt or the
        conduit_adjset_validate tool.
 */
void save_mesh(const conduit::Node &root, const std::string &filebase)
{
    // NOTE: Enable this to write files for debugging.
#if 0
    const std::string protocol("hdf5");
    conduit::relay::io::blueprint::save_mesh(root, filebase, protocol);
#else
    std::cout << "Skip writing " << filebase << std::endl;
#endif
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, shapetype)
{
    struct test_case
    {
        std::string meshtype;
        std::string shapetype;
        int sdim;
        // answers
        int tdim;
        bool is_poly, is_polygonal, is_polyhedral;
    };
    std::vector<test_case> tests{
        // non-unstructured topo types.
        {"points", "point", 2, 0, false, false, false},
        {"points", "point", 3, 0, false, false, false},
        {"points_implicit", "point", 2, 0, false, false, false},
        {"points_implicit", "point", 3, 0, false, false, false},

        {"uniform", "line", 1, 1, false, false, false},
        {"uniform", "quad", 2, 2, false, false, false},
        {"uniform", "hex", 3, 3, false, false, false},

        {"rectilinear", "line", 1, 1, false, false, false},
        {"rectilinear", "quad", 2, 2, false, false, false},
        {"rectilinear", "hex", 3, 3, false, false, false},

        {"structured", "quad", 2, 2, false, false, false},
        {"structured", "hex", 3, 3, false, false, false},

        // unstructured types
        {"lines", "line", 2, 1, false, false, false},
        {"lines", "line", 3, 1, false, false, false},

        {"tris", "tri", 2, 2, false, false, false},

        {"quads", "quad", 2, 2, false, false, false},
        {"quads_poly", "polygonal", 2, 2, true, true, false},

        {"tets", "tet", 3, 3, false, false, false},

        {"pyramids", "pyramid", 3, 3, false, false, false},
        {"wedges", "wedge", 3, 3, false, false, false},

        {"hexs", "hex", 3, 3, false, false, false},
        {"hexs_poly", "polyhedral", 3, 3, true, false, true}
    };

    for(const auto &t : tests)
    {
        const conduit::index_t s = 5;
        conduit::Node mesh;
        conduit::blueprint::mesh::examples::braid(t.meshtype,
            s, t.sdim >= 2 ? s : 0, t.sdim >= 3 ? s : 0,
            mesh);

        // Make sure we can pass various topology types to ShapeType.
        const conduit::Node &topo = mesh["topologies/mesh"];
        conduit::blueprint::mesh::utils::ShapeType shape(topo);

        EXPECT_EQ(conduit::blueprint::mesh::coordset::dims(mesh["coordsets/coords"]), t.sdim);
        EXPECT_EQ(shape.type, t.shapetype);
        EXPECT_EQ(shape.dim, t.tdim);
        EXPECT_EQ(shape.is_poly(), t.is_poly);
        EXPECT_EQ(shape.is_polygonal(), t.is_polygonal);
        EXPECT_EQ(shape.is_polyhedral(), t.is_polyhedral);
    }

    // Initialize with a bad shape name.
    conduit::blueprint::mesh::utils::ShapeType shape("bogus");
    EXPECT_FALSE(shape.is_poly());
    EXPECT_FALSE(shape.is_polygonal());
    EXPECT_FALSE(shape.is_polyhedral());
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, adjset_validate_element_0d)
{
    conduit::Node root, info;
    create_2_domain_0d_mesh(root, 0, 1);
    save_mesh(root, "adjset_validate_element_0d");
    bool res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_TRUE(res);

    // Now, adjust the adjsets so they are wrong on both domains.
    root["domain0/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0});
    root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{2});
    info.reset();
    save_mesh(root, "adjset_validate_element_0d_bad");
    res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    EXPECT_TRUE(info.has_path("domain0/main_adjset/domain0_1"));
    const conduit::Node &n0 = info["domain0/main_adjset/domain0_1"];
    EXPECT_EQ(n0.number_of_children(), 1);
    const conduit::Node &c0 = n0[0];
    EXPECT_TRUE(c0.has_path("element"));
    EXPECT_TRUE(c0.has_path("neighbor"));
    EXPECT_EQ(c0["element"].to_int(), 0);
    EXPECT_EQ(c0["neighbor"].to_int(), 1);

    EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
    const conduit::Node &n1 = info["domain1/main_adjset/domain0_1"];
    EXPECT_EQ(n1.number_of_children(), 1);
    const conduit::Node &c1 = n1[0];
    EXPECT_TRUE(c1.has_path("element"));
    EXPECT_TRUE(c1.has_path("neighbor"));
    EXPECT_EQ(c1["element"].to_int(), 2);
    EXPECT_EQ(c1["neighbor"].to_int(), 0);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, adjset_validate_element_1d)
{
    conduit::Node root, info;
    create_2_domain_1d_mesh(root, 0, 1);
    save_mesh(root, "adjset_validate_element_1d");
    bool res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_TRUE(res);

    // Now, adjust the adjsets so they are wrong on both domains.
    root["domain0/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0});
    root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{1});
    info.reset();
    save_mesh(root, "adjset_validate_element_1d_bad");
    res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    EXPECT_TRUE(info.has_path("domain0/main_adjset/domain0_1"));
    const conduit::Node &n0 = info["domain0/main_adjset/domain0_1"];
    EXPECT_EQ(n0.number_of_children(), 1);
    const conduit::Node &c0 = n0[0];
    EXPECT_TRUE(c0.has_path("element"));
    EXPECT_TRUE(c0.has_path("neighbor"));
    EXPECT_EQ(c0["element"].to_int(), 0);
    EXPECT_EQ(c0["neighbor"].to_int(), 1);

    EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
    const conduit::Node &n1 = info["domain1/main_adjset/domain0_1"];
    EXPECT_EQ(n1.number_of_children(), 1);
    const conduit::Node &c1 = n1[0];
    EXPECT_TRUE(c1.has_path("element"));
    EXPECT_TRUE(c1.has_path("neighbor"));
    EXPECT_EQ(c1["element"].to_int(), 1);
    EXPECT_EQ(c1["neighbor"].to_int(), 0);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, adjset_validate_element_2d)
{
    conduit::Node root, info;
    create_2_domain_2d_mesh(root, 0, 1);
    save_mesh(root, "adjset_validate_element_2d");
    bool res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_TRUE(res);
    //info.print();

    // Now, adjust the adjset for domain1 so it includes an element not present in domain 0
    root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0,2,4});
    info.reset();
    save_mesh(root, "adjset_validate_element_2d_bad");
    res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
    const conduit::Node &n = info["domain1/main_adjset/domain0_1"];
    EXPECT_EQ(n.number_of_children(), 1);
    const conduit::Node &c = n[0];
    EXPECT_TRUE(c.has_path("element"));
    EXPECT_TRUE(c.has_path("neighbor"));

    EXPECT_EQ(c["element"].to_int(), 2);
    EXPECT_EQ(c["neighbor"].to_int(), 0);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, adjset_validate_element_3d)
{
    conduit::Node root, info;
    create_2_domain_3d_mesh(root, 0, 1);
    save_mesh(root, "adjset_validate_element_3d");
    bool res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_TRUE(res);

    // Now, adjust the adjsets so they are wrong on both domains.
    root["domain0/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{0});
    root["domain1/adjsets/main_adjset/groups/domain0_1/values"].set(std::vector<int>{2});
    info.reset();
    save_mesh(root, "adjset_validate_element_3d_bad");
    res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_FALSE(res);
    //info.print();

    EXPECT_TRUE(info.has_path("domain0/main_adjset/domain0_1"));
    const conduit::Node &n0 = info["domain0/main_adjset/domain0_1"];
    EXPECT_EQ(n0.number_of_children(), 1);
    const conduit::Node &c0 = n0[0];
    EXPECT_TRUE(c0.has_path("element"));
    EXPECT_TRUE(c0.has_path("neighbor"));
    EXPECT_EQ(c0["element"].to_int(), 0);
    EXPECT_EQ(c0["neighbor"].to_int(), 1);

    EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
    const conduit::Node &n1 = info["domain1/main_adjset/domain0_1"];
    EXPECT_EQ(n1.number_of_children(), 1);
    const conduit::Node &c1 = n1[0];
    EXPECT_TRUE(c1.has_path("element"));
    EXPECT_TRUE(c1.has_path("neighbor"));
    EXPECT_EQ(c1["element"].to_int(), 2);
    EXPECT_EQ(c1["neighbor"].to_int(), 0);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, adjset_validate_vertex_3d)
{
    conduit::Node root, info;
    create_2_domain_3d_mesh(root, 0, 1);
    // Add adjsets
    conduit::Node &d0_adjset = root["domain0/adjsets/main_adjset"];
    d0_adjset["association"] = "vertex";
    d0_adjset["topology"] = "main";
    conduit::Node &d0_01 = d0_adjset["groups/domain0_1"];
    d0_01["neighbors"] = 1;
    d0_01["values"].set(std::vector<int>{1,2,3,5,6,7,9,10,11,13,14,15});

    conduit::Node &d1_adjset = root["domain1/adjsets/main_adjset"];
    d1_adjset["association"] = "vertex";
    d1_adjset["topology"] = "main";
    conduit::Node &d1_01 = d1_adjset["groups/domain0_1"];
    d1_01["neighbors"] = 0;
    d1_01["values"].set(std::vector<int>{0,1,2,4,5,6,8,9,10,12,13,14});

    EXPECT_TRUE(conduit::blueprint::mesh::adjset::verify(d0_adjset, info));
    info.reset();
    EXPECT_TRUE(conduit::blueprint::mesh::adjset::verify(d1_adjset, info));

    save_mesh(root, "adjset_validate_vertex_3d");
    bool res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    EXPECT_TRUE(res);

    // Now, adjust the adjsets so they are wrong on both domains.
    d0_01["values"].set(std::vector<int>{1,2,3,5,6,7,9,10,11,13,14,15,/*wrong*/0,4,8,12});
    d1_01["values"].set(std::vector<int>{0,1,2,4,5,6,8,9,10,12,13,14,/*wrong*/3,7,11,15});
    info.reset();
    save_mesh(root, "adjset_validate_vertex_3d_bad");
    res = conduit::blueprint::mesh::utils::adjset::validate(root, "main_adjset", info);
    //info.print();
    EXPECT_FALSE(res);

    EXPECT_TRUE(info.has_path("domain0/main_adjset/domain0_1"));
    const conduit::Node &n0 = info["domain0/main_adjset/domain0_1"];
    EXPECT_EQ(n0.number_of_children(), 4);
    const std::vector<int> d0err_vertex{0,4,8,12};
    for(conduit::index_t i = 0; i < 4; i++)
    {
        EXPECT_EQ(n0[i]["neighbor"].to_int(), 1);
        EXPECT_TRUE(std::find(d0err_vertex.begin(), d0err_vertex.end(), n0[i]["vertex"].to_int()) != d0err_vertex.end());
    }

    EXPECT_TRUE(info.has_path("domain1/main_adjset/domain0_1"));
    const conduit::Node &n1 = info["domain1/main_adjset/domain0_1"];
    EXPECT_EQ(n1.number_of_children(), 4);
    const std::vector<int> d1err_vertex{3,7,11,15};
    for(conduit::index_t i = 0; i < 4; i++)
    {
        EXPECT_EQ(n1[i]["neighbor"].to_int(), 0);
        EXPECT_TRUE(std::find(d1err_vertex.begin(), d1err_vertex.end(), n1[i]["vertex"].to_int()) != d1err_vertex.end());
    }
}

//-----------------------------------------------------------------------------
template <typename CoordType, typename ConnType>
void test_rewrite_connectivity(int dims)
{
    conduit::Node n;
    n["coordsets/coords1/type"] = "explicit";
    n["coordsets/coords1/values/x"].set(std::vector<CoordType>{0.f,1.f,2.f,0.f,1.f,2.f,0.f,1.f,2.f});
    n["coordsets/coords1/values/y"].set(std::vector<CoordType>{0.f,0.f,0.f,1.f,1.f,1.f,2.f,2.f,2.f});
    if(dims > 2)
        n["coordsets/coords1/values/z"].set(std::vector<CoordType>{5.f,5.f,5.f,5.f,5.f,5.f,5.f,5.f,5.f});
    n["topologies/mesh1/type"] = "unstructured";
    n["topologies/mesh1/coordset"] = "coords1";
    n["topologies/mesh1/elements/shape"] = "quad";
    n["topologies/mesh1/elements/connectivity"].set(std::vector<ConnType>{0,1,4,3, 1,2,5,4, 3,4,7,6, 4,5,8,7});
    n["topologies/mesh1/elements/sizes"].set(std::vector<ConnType>{4,4,4,4});
    n["topologies/mesh1/elements/offsets"].set(std::vector<ConnType>{0,4,8,12});

    n["coordsets/coords2/type"] = "explicit";
    n["coordsets/coords2/values/x"].set(std::vector<CoordType>{1.f,2.f,1.f,2.f});
    n["coordsets/coords2/values/y"].set(std::vector<CoordType>{1.f,1.f,2.f,2.f});
    if(dims > 2)
        n["coordsets/coords2/values/z"].set(std::vector<CoordType>{5.f,5.f,5.f,5.f});
    n["topologies/mesh2/type"] = "unstructured";
    n["topologies/mesh2/coordset"] = "coords2";
    n["topologies/mesh2/elements/shape"] = "tri";
    n["topologies/mesh2/elements/connectivity"].set(std::vector<ConnType>{0,1,2, 1,3,2});
    n["topologies/mesh2/elements/sizes"].set(std::vector<ConnType>{3,3});
    n["topologies/mesh2/elements/offsets"].set(std::vector<ConnType>{0,3});

    conduit::Node info;
    EXPECT_TRUE(conduit::blueprint::mesh::topology::verify(n["topologies/mesh1"], info));
    EXPECT_TRUE(conduit::blueprint::mesh::topology::verify(n["topologies/mesh2"], info));

    // Make mesh2 use coords1
    conduit::blueprint::mesh::utils::topology::unstructured::rewrite_connectivity(n["topologies/mesh2"],
                                                                                  n["coordsets/coords1"]);

    // Make sure that mesh2's connectivity uses coords1 ids.
    auto conn = n["topologies/mesh2/elements/connectivity"].as_int_accessor();
    EXPECT_EQ(conn[0], 4);
    EXPECT_EQ(conn[1], 5);
    EXPECT_EQ(conn[2], 7);

    EXPECT_EQ(conn[3], 5);
    EXPECT_EQ(conn[4], 8);
    EXPECT_EQ(conn[5], 7);

    EXPECT_EQ(n["topologies/mesh2/coordset"].as_string(), "coords1");
}

void test_rewrite_connectivity_top(int dims)
{
    test_rewrite_connectivity<float, conduit::int32>(dims);
    test_rewrite_connectivity<float, conduit::uint32>(dims);
    test_rewrite_connectivity<float, conduit::int64>(dims);
    test_rewrite_connectivity<float, conduit::uint64>(dims);
    test_rewrite_connectivity<float, conduit::index_t>(dims);

    test_rewrite_connectivity<double, conduit::int32>(dims);
    test_rewrite_connectivity<double, conduit::uint32>(dims);
    test_rewrite_connectivity<double, conduit::int64>(dims);
    test_rewrite_connectivity<double, conduit::uint64>(dims);
    test_rewrite_connectivity<double, conduit::index_t>(dims);
}

TEST(conduit_blueprint_mesh_utils, rewrite_connectivity_2d)
{
    test_rewrite_connectivity_top(2);
}

TEST(conduit_blueprint_mesh_utils, rewrite_connectivity_3d)
{
    test_rewrite_connectivity_top(3);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, copy_fields)
{
    conduit::Node n;
    n["fields/f1/association"] = "vertex";
    n["fields/f1/topology"] = "mesh1";
    n["fields/f1/values"].set(std::vector<double>{0., 1., 2., 3.});
    n["fields/f2/association"] = "vertex";
    n["fields/f2/topology"] = "mesh1";
    n["fields/f2/values"].set(std::vector<double>{4., 5., 6., 7.});
    n["fields/f3/association"] = "vertex";
    n["fields/f3/topology"] = "mesh2";
    n["fields/f3/values"].set(std::vector<double>{8., 9., 10., 11.});
    n["fields/f4/association"] = "vertex";
    n["fields/f4/topology"] = "mesh2";
    n["fields/f4/values"].set(std::vector<double>{12., 13., 14., 15.});

    conduit::Node opts, fields;
    opts["exclusions"].append().set("f2");
    conduit::blueprint::mesh::utils::copy_fields(n["fields"], fields, opts);
    EXPECT_EQ(fields.number_of_children(), 3);
    EXPECT_EQ(fields[0].name(), "f1");
    EXPECT_EQ(fields[1].name(), "f3");
    EXPECT_EQ(fields[2].name(), "f4");

    opts.reset();
    fields.reset();
    opts["topology"] = "mesh2";
    conduit::blueprint::mesh::utils::copy_fields(n["fields"], fields, opts);
    EXPECT_EQ(fields.number_of_children(), 2);
    EXPECT_EQ(fields[0].name(), "f3");
    EXPECT_EQ(fields[1].name(), "f4");

    fields.reset();
    opts["exclusions"].append().set("f3");
    conduit::blueprint::mesh::utils::copy_fields(n["fields"], fields, opts);
    EXPECT_EQ(fields.number_of_children(), 1);
    EXPECT_EQ(fields[0].name(), "f4");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, adjset_compare_pointwise_2d)
{
    conduit::Node root, info;
    create_2_domain_2d_mesh(root, 0, 1);
    save_mesh(root, "adjset_compare_pointwise_2d");
    auto domains = conduit::blueprint::mesh::domains(root);

    for(const auto &domPtr : domains)
    {
        // It's not in canonical form.
        const conduit::Node &adjset = domPtr->fetch_existing("adjsets/pt_adjset");
        bool canonical = conduit::blueprint::mesh::utils::adjset::is_canonical(adjset);
        EXPECT_FALSE(canonical);

        // The fails_pointwise adjset is in canonical form.
        const conduit::Node &adjset2 = domPtr->fetch_existing("adjsets/fails_pointwise");
        canonical = conduit::blueprint::mesh::utils::adjset::is_canonical(adjset2);
        EXPECT_TRUE(canonical);
    }

    // Check that we can still run compare_pointwise - it will convert internally.
    bool eq = conduit::blueprint::mesh::utils::adjset::compare_pointwise(root, "pt_adjset", info);
    if(!eq)
        info.print();
    EXPECT_TRUE(eq);

    // Make sure the extra adjset was removed.
    for(const auto &domPtr : domains)
    {
        // It's not in canonical form.
        bool tmpExists = domPtr->has_path("adjsets/__pt_adjset__");
        EXPECT_FALSE(tmpExists);
    }

    // Force it to be canonical
    for(const auto &domPtr : domains)
    {
        // It's not in canonical form.
        conduit::Node &adjset = domPtr->fetch_existing("adjsets/pt_adjset");
        conduit::blueprint::mesh::utils::adjset::canonicalize(adjset);
    }
    info.reset();
    eq = conduit::blueprint::mesh::utils::adjset::compare_pointwise(root, "pt_adjset", info);
    if(!eq)
       info.print();
    EXPECT_TRUE(eq);

    // Test that the fails_pointwise adjset actually fails.
    info.reset();
    eq = conduit::blueprint::mesh::utils::adjset::compare_pointwise(root, "fails_pointwise", info);
    //if(!eq)
    //   info.print();
    EXPECT_FALSE(eq);

    // Test that the notevenclose adjset actually fails.
    info.reset();
    eq = conduit::blueprint::mesh::utils::adjset::compare_pointwise(root, "notevenclose", info);
    //if(!eq)
    //   info.print();
    EXPECT_FALSE(eq);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, topology_search_2d)
{
    conduit::Node root, info;
    create_2_domain_2d_mesh(root, 0, 1);
    save_mesh(root, "topology_search_2d");
    auto domains = conduit::blueprint::mesh::domains(root);

    // Look for domain 0 in domain 1. They should intersect by 2 zones.
    auto res01 = conduit::blueprint::mesh::utils::topology::search(
                   domains[1]->fetch_existing("topologies/main"),
                   domains[0]->fetch_existing("topologies/main"));
    const std::vector<int> answers01{0,0,1,0,0,0,1};
    EXPECT_EQ(res01.size(), answers01.size());
    for(size_t i = 0; i < res01.size(); i++)
    {
        EXPECT_EQ(res01[i], answers01[i]);
    }

    // Look for domain 1 in domain 0. They should intersect by 2 zones.
    auto res10 = conduit::blueprint::mesh::utils::topology::search(
                   domains[0]->fetch_existing("topologies/main"),
                   domains[1]->fetch_existing("topologies/main"));
    const std::vector<int> answers10{1,0,0,0,1,0};
    EXPECT_EQ(res10.size(), answers10.size());
    for(size_t i = 0; i < res10.size(); i++)
    {
        EXPECT_EQ(res10[i], answers10[i]);
    }

    // Turn the main_adjset adjset into a topology.
    conduit::Node main_adjset_topo;
    conduit::blueprint::mesh::utils::adjset::to_topo(root, "main_adjset", main_adjset_topo);
    //main_adjset_topo.print();

    // Pull domain 0 adjset zones out as a topology and look for them in domain 1.
    const conduit::Node &t01 = main_adjset_topo.fetch_existing("topologies/main_adjset_0_group_0_1");
    EXPECT_EQ(t01["type"].as_string(), "unstructured");
    EXPECT_EQ(t01["elements/shape"].as_string(), "quad");
    EXPECT_EQ(conduit::blueprint::mesh::topology::length(t01), 2);
    auto resT01 = conduit::blueprint::mesh::utils::topology::search(
                      domains[1]->fetch_existing("topologies/main"),
                      t01);
    const std::vector<int> answersT01{1,1};
    EXPECT_EQ(resT01.size(), answersT01.size());
    for(size_t i = 0; i < resT01.size(); i++)
    {
        EXPECT_EQ(resT01[i], answersT01[i]);
    }

    // Pull domain 1 adjset zones out as a topology and look for them in domain 0.
    const conduit::Node &t10 = main_adjset_topo.fetch_existing("topologies/main_adjset_1_group_0_1");
    EXPECT_EQ(t10["type"].as_string(), "unstructured");
    EXPECT_EQ(t10["elements/shape"].as_string(), "quad");
    EXPECT_EQ(conduit::blueprint::mesh::topology::length(t10), 2);
    auto resT10 = conduit::blueprint::mesh::utils::topology::search(
                      domains[0]->fetch_existing("topologies/main"),
                      t10);
    const std::vector<int> answersT10{1,1};
    EXPECT_EQ(resT10.size(), answersT10.size());
    for(size_t i = 0; i < resT10.size(); i++)
    {
        EXPECT_EQ(resT10[i], answersT10[i]);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, topology_search_2d_nohit)
{
    // A consists of 3 quads in the 1st column
    // B is a single quad on the right
    // C consists of 4 tris on the lower left.
    //
    // A and B do not overlap.
    // A and C partially overlap but are different types.
    const char *example = R"(
coordsets:
  coords:
    type: explicit
    values:
      x: [0.,1.,2.,3.,0.,1.,2.,3.,0.,1.,2.,3.,0.,1.,2.,3.]
      y: [0.,0.,0.,0.,1.,1.,1.,1.,2.,2.,2.,2.,3.,3.,3.,3.]
topologies:
  A:
    type: unstructured
    coordset: coords
    elements:
      shape: quad
      connectivity: [0,1,5,4,4,5,9,8,8,9,13,12]
      offsets: [0,4,8]
  B:
    type: unstructured
    coordset: coords
    elements:
      shape: quad
      connectivity: [6,7,11,10]
      offsets: [0]
  C:
    type: unstructured
    coordset: coords
    elements:
      shape: tri
      connectivity: [0,1,4,1,5,4,1,2,5,2,6,5]
      offsets: [0,3,6,9]
)";

    conduit::Node n;
    n.parse(example, "yaml");

    // Look for B in A
    auto resBA = conduit::blueprint::mesh::utils::topology::search(
                   n.fetch_existing("topologies/A"),
                   n.fetch_existing("topologies/B"));
    const std::vector<int> answers01{0,0,1,0,0,0,1};
    EXPECT_EQ(resBA.size(), 1);
    EXPECT_EQ(resBA[0], 0);

    // Look for C in A
    auto resCA = conduit::blueprint::mesh::utils::topology::search(
                   n.fetch_existing("topologies/A"),
                   n.fetch_existing("topologies/C"));
    const std::vector<int> answersCA{0,0,0,0};
    EXPECT_EQ(resCA.size(), 4);
    EXPECT_EQ(resCA, answersCA);
};

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, is_canonical)
{
    const char *adjsets = R"(
adjsets:
  correct:
    topology: mesh
    association: vertex
    groups:
      group_0_1:
        neighbors: 1
        values: [0,1,2]
  wrong_prefix:
    topology: mesh
    association: vertex
    groups:
      prefix_0_1:
        neighbors: 1
        values: [0,1,2]
  out_of_order:
    topology: mesh
    association: vertex
    groups:
      group_1_0:
        neighbors: 1
        values: [0,1,2]
  multi:
    topology: mesh
    association: vertex
    groups:
      group_0_1_2:
        neighbors: [1,2]
        values: [0,1,2]
  multi_out_of_order:
    topology: mesh
    association: vertex
    groups:
      group_0_2_1:
        neighbors: [1,2]
        values: [0,1,2]
)";

    conduit::Node n;
    n.parse(adjsets);
    EXPECT_TRUE( conduit::blueprint::mesh::utils::adjset::is_canonical(n["adjsets/correct"]));
    EXPECT_FALSE(conduit::blueprint::mesh::utils::adjset::is_canonical(n["adjsets/wrong_prefix"]));
    EXPECT_FALSE(conduit::blueprint::mesh::utils::adjset::is_canonical(n["adjsets/out_of_order"]));
    EXPECT_TRUE( conduit::blueprint::mesh::utils::adjset::is_canonical(n["adjsets/multi"]));
    EXPECT_FALSE(conduit::blueprint::mesh::utils::adjset::is_canonical(n["adjsets/multi_out_of_order"]));
}
