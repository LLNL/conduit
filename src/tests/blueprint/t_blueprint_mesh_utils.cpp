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

#include <algorithm>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;
namespace bputils = conduit::blueprint::mesh::utils;

//---------------------------------------------------------------------------
void save_mesh(const conduit::Node &root, const std::string &filebase)
{
    const std::string protocol("hdf5");
    conduit::relay::io::blueprint::save_mesh(root, filebase, protocol);
}

//---------------------------------------------------------------------------
void create_2_domain_0d_mesh(conduit::Node &root)
{
    // The adjset is properly set up.
    //
    // dom0 *       *       *
    // dom1         *       *       *
    const char *example = R"(
domain0:
  state:
    domain_id: 0
  coordsets:
    coords:
      type: explicit
      values:
        x: [0.,1.,2.]
        y: [0.,0.,0.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: point
        connectivity: [0,1,2]
        offsets: [0,1,2]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 1
          values: [1,2]
domain1:
  state:
    domain_id: 1
  coordsets:
    coords:
      type: explicit
      values:
        x: [1.,2.,3.]
        y: [0.,0.,0.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: point
        connectivity: [0,1,2]
        offsets: [0,1,2]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 0
          values: [0,1]
)";

    root.parse(example, "yaml");
}

//---------------------------------------------------------------------------
void create_2_domain_1d_mesh(conduit::Node &root)
{
    // The adjset is properly set up.
    //
    // dom0 *-------*-------*
    // dom1         *-------*-------*
    const char *example = R"(
domain0:
  state:
    domain_id: 0
  coordsets:
    coords:
      type: explicit
      values:
        x: [0.,1.,2.]
        y: [0.,0.,0.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: line
        connectivity: [0,1,1,2]
        offsets: [0,2]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 1
          values: 1
domain1:
  state:
    domain_id: 1
  coordsets:
    coords:
      type: explicit
      values:
        x: [1.,2.,3.]
        y: [0.,0.,0.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: line
        connectivity: [0,1,1,2]
        offsets: [0,2]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 0
          values: 0
)";

    root.parse(example, "yaml");
}

//---------------------------------------------------------------------------
void create_2_domain_2d_mesh(conduit::Node &root)
{
    // The adjset is properly set up
    const char *example = R"(
domain0:
  state:
    domain_id: 0
  coordsets:
    coords:
      type: explicit
      values:
        x: [0.,1.,2.,3.,0.,1.,2.,3.,0.,1.,2.,3.,0.,1.,2.,3.]
        y: [0.,0.,0.,0.,1.,1.,1.,1.,2.,2.,2.,2.,3.,3.,3.,3.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: quad
        connectivity: [0,1,5,4,1,2,6,5,2,3,7,6,4,5,9,8,8,9,13,12,9,10,14,13,10,11,15,14]
        offsets: [0,4,8,12,16,20,24]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 1
          values: [2, 6]
domain1:
  state:
    domain_id: 1
  coordsets:
    coords:
      type: explicit
      values:
        x: [2.,3.,4.,2.,3.,4.,2.,3.,4.,2.,3.,4.]
        y: [0.,0.,0.,1.,1.,1.,2.,2.,2.,3.,3.,3.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: quad
        connectivity: [0,1,4,3,1,2,5,4,3,4,7,6,4,5,8,7,6,7,10,9,7,8,11,10]
        offsets: [0,4,8,12,16,20]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 0
          values: [0,4]
)";

    root.parse(example, "yaml");
}

//---------------------------------------------------------------------------
void create_2_domain_3d_mesh(conduit::Node &root)
{
    // The adjset is properly set up.
    //
    // dom0 *-------*-------*-------*
    // dom1         *-------*-------*-------*
    const char *example = R"(
domain0:
  state:
    domain_id: 0
  coordsets:
    coords:
      type: explicit
      values:
        x: [0.,1.,2.,3.,0.,1.,2.,3.,0.,1.,2.,3.,0.,1.,2.,3.]
        y: [0.,0.,0.,0.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.]
        z: [0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: hex
        connectivity: [0,1,5,4,8,9,13,12,1,2,6,5,9,10,14,13,2,3,7,6,10,11,15,14]
        offsets: [0,8,16]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 1
          values: [1,2]
domain1:
  state:
    domain_id: 1
  coordsets:
    coords:
      type: explicit
      values:
        x: [1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.,1.,2.,3.,4.]
        y: [0.,0.,0.,0.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.]
        z: [0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.]
  topologies:
    main:
      type: unstructured
      coordset: coords
      elements:
        shape: hex
        connectivity: [0,1,5,4,8,9,13,12,1,2,6,5,9,10,14,13,2,3,7,6,10,11,15,14]
        offsets: [0,8,16]
  adjsets:
    main_adjset:
      association: element
      topology: main
      groups:
        domain0_1:
          neighbors: 0
          values: [0,1]
)";

    root.parse(example, "yaml");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_utils, adjset_validate_element_0d)
{
    conduit::Node root, info;
    create_2_domain_0d_mesh(root);
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
    info.print();

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
    create_2_domain_1d_mesh(root);
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
    info.print();

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
    create_2_domain_2d_mesh(root);
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
    info.print();

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
    create_2_domain_3d_mesh(root);
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
    info.print();

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
