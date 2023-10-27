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
    info.print();

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
