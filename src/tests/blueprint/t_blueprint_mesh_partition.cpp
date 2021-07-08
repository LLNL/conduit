// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_partition.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <math.h>
#include <iostream>
#include "gtest/gtest.h"

using std::cout;
using std::endl;

void
tmp_err_handler(const std::string &s1, const std::string &s2, int i1)
{
    cout << "s1=" << s1 << ", s2=" << s2 << ", i1=" << i1 << endl;

    while(1);
}
//-----------------------------------------------------------------------------
void
test_logical_selection_2d(const std::string &topo)
{
    conduit::utils::set_error_handler(tmp_err_handler);

    // Make 10x10x1 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {11,11,1};
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
 
    // With no options, test that output==input
    conduit::blueprint::mesh::partition(input, options, output);
    cout << "output=";
    output.print();
    EXPECT_EQ(input.diff(output, msg, 0.0), false);
}

//-----------------------------------------------------------------------------
void
test_logical_selection_3d(const std::string &topo)
{
    conduit::utils::set_error_handler(tmp_err_handler);


    // Make 10x10x10 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {11,11,11};
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
 
    // With no options, test that output==input
    conduit::blueprint::mesh::partition(input, options, output);
    cout << "output=";
    output.print();
    EXPECT_EQ(input.diff(output, msg, 0.0), false);
#if 0
    // Select the whole thing but divide it into target domains.
    const char *opt1 =
"target: 4";
    options.parse(opt1, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(output.number_of_children(), 4);
    EXPECT_EQ(conduit::blueprint::mesh::is_multi_domain(output), true);
    // TODO: test output mesh contents.
    output.print();

    // Select a portion of the input mesh cells using options.
    const char *opt2 =
"selections:\n"
"   -\n"
"     type: \"logical\"\n"
"     domain: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,4,4]\n";
    options.parse(opt2, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
//    EXPECT_EQ(conduit::blueprint::mesh::utils::topology::length(output[0]), 5*5*5);
    output.print();
    // TODO: test output mesh contents.

    // Select a larger portion of the input mesh using options but split into
    // multiple domains.
    const char *opt3 =
"selections:\n"
"   -\n"
"     type: \"logical\"\n"
"     domain: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,9,9]\n"
"target: 8\n";
    options.parse(opt3, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::utils::topology::length(output[0]), 5*5*5);
    output.print();
    // TODO: test output mesh contents.

    // Select 2 logical subsets of the input mesh using options.
    //   - expect 2 output domains.
    const char *opt4 =
"selections:\n"
"   -\n"
"     type: \"logical\"\n"
"     domain: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,4,4]\n"
"   -\n"
"     type: \"logical\"\n"
"     domain: 0\n"
"     start: [5,0,0]\n"
"     end:   [9,4,4]\n";
    options.parse(opt2, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    output.print();
    EXPECT_EQ(output.number_of_children(), 2);
    EXPECT_EQ(conduit::blueprint::mesh::is_multi_domain(output), true);
// These fail right now.
//    EXPECT_EQ(output[0]["type"].as_string(), topo);
//    EXPECT_EQ(output[1]["type"].as_string(), topo);
    EXPECT_EQ(conduit::blueprint::mesh::utils::topology::length(output[0]), 5*5*5);
    EXPECT_EQ(conduit::blueprint::mesh::utils::topology::length(output[1]), 5*5*5);

    // Select 2 logical subsets of the input mesh using options but turn them
    // into 1 domain.
    //  - expect 1 domain
    const char *opt5 =
"selections:\n"
"   -\n"
"     type: \"logical\"\n"
"     domain: 0\n"
"     start: [0,0,0]\n"
"     end:   [9,7,4]\n"
"   -\n"
"     type: \"logical\"\n"
"     domain: 0\n"
"     start: [0,0,5]\n"
"     end:   [9,7,9]\n"
"target: 1\n";

    options.parse(opt5, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    output.print();
    EXPECT_EQ(output.number_of_children(), 1);
    EXPECT_EQ(conduit::blueprint::mesh::is_multi_domain(output), false);
// These fail right now.
//    EXPECT_EQ(output[0]["type"].as_string(), topo);
//    EXPECT_EQ(output[1]["type"].as_string(), topo);
//    EXPECT_EQ(conduit::blueprint::mesh::utils::topology::length(output[0]), 10*8*5);
//    EXPECT_EQ(conduit::blueprint::mesh::utils::topology::length(output[1]), 10*8*5);


    // TODO: Make a multi domain mesh and test pulling out multiple logical
    //       selections from it.


    // TODO: Make a multi domain mesh and test pulling out multiple logical
    //       selections from it and set target to 1 or 2 so we trigger combining.
#endif
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_logical_2d)
{
    test_logical_selection_2d("uniform");
}

#if 0
//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_logical_3d)
{
    test_logical_selection_3d("uniform");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_rectilinear)
{
    test_logical_selection("rectilinear");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_structured)
{
    test_logical_selection("structured");
}
#endif
