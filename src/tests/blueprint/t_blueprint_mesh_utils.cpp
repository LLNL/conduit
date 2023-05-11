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
