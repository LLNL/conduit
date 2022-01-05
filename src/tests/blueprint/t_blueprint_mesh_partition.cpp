// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_partition.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mesh_partition.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <math.h>
#include <iostream>
#include <array>
#include <cmath>
#include "gtest/gtest.h"

#include "blueprint_test_helpers.hpp"

using std::cout;
using std::endl;

// Enable this macro to generate baselines.
// #define GENERATE_BASELINES

// #define USE_ERROR_HANDLER

#ifndef ALWAYS_PRINT
static const bool always_print = false;
#else
static const bool always_print = true;
#endif

//-----------------------------------------------------------------------------
#ifdef _WIN32
const std::string sep("\\");
#else
const std::string sep("/");
#endif

//-----------------------------------------------------------------------------
std::string
baseline_dir()
{
    std::string path(__FILE__);
    auto idx = path.rfind(sep);
    if(idx != std::string::npos)
        path = path.substr(0, idx);
    path = path + sep + std::string("baselines");
    return path;
}

//-----------------------------------------------------------------------------
std::string test_name() { return std::string("t_blueprint_mesh_partition"); }

//-----------------------------------------------------------------------------
int get_rank() { return 0; }

//-----------------------------------------------------------------------------
void barrier() { }

//-----------------------------------------------------------------------------
// Include some helper function definitions
#include "blueprint_baseline_helpers.hpp"

#if 0
//-----------------------------------------------------------------------------
void
tmp_err_handler(const std::string &s1, const std::string &s2, int i1)
{
    cout << "s1=" << s1 << ", s2=" << s2 << ", i1=" << i1 << endl;

    while(1);
}

//-----------------------------------------------------------------------------
void
test_logical_selection_2d(const std::string &topo, const std::string &base)
{
#ifdef USE_ERROR_HANDLER
    conduit::utils::set_error_handler(tmp_err_handler);
#endif

    // Make 10x10x1 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {11,11,1};
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
    // Override with int64 because YAML loses int/uint information.
    conduit::int64 i100 = 100;
    input["state/cycle"].set(i100);
    input["state/domain_id"].set((int)0);

    // With no options (turn mapping off though because otherwise we add 
    // the original vertex and element fields), test that output==input
    const char *opt0 =
"mapping: 0";
    options.reset(); options.parse(opt0, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(input.diff(output, msg, 0.0, true), false) << msg.to_json();
    std::string b00 = baseline_file(base + "_00");
    save_visit(b00, output);

    // Select the whole thing but divide it into target domains.
    const char *opt1 =
"target: 2";
    options.reset(); options.parse(opt1, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 2);
    std::string b01 = baseline_file(base + "_01");
    save_visit(b01, output);
#ifdef GENERATE_BASELINES
    make_baseline(b01, output);
#else
    EXPECT_EQ(compare_baseline(b01, output), true);
#endif

    // Select the whole thing but divide it into target domains.
    const char *opt2 =
"target: 4";
    options.reset(); options.parse(opt2, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 4);
    std::string b02 = baseline_file(base + "_02");
    save_visit(b02, output);
#ifdef GENERATE_BASELINES
    make_baseline(b02, output);
#else
    EXPECT_EQ(compare_baseline(b02, output), true);
#endif

    // Select the whole thing but go out of bounds to see if the selections
    // clamp to good values.
    const char *opt3 =
"selections:\n"
"   -\n"
"     type: \"logical\"\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [100,100,100]";
    options.reset(); options.parse(opt3, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    EXPECT_EQ(conduit::blueprint::mesh::is_multi_domain(output), false);
    std::string b03 = baseline_file(base + "_03");
    save_visit(b03, output);
#ifdef GENERATE_BASELINES
    make_baseline(b03, output);
#else
    EXPECT_EQ(compare_baseline(b03, output), true);
#endif

    // Select 3 parts.
    const char *opt4 =
"selections:\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,9,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,0,0]\n"
"     end:   [9,4,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,5,0]\n"
"     end:   [9,9,0]";
    options.reset(); options.parse(opt4, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 3);
    std::string b04 = baseline_file(base + "_04");
    save_visit(b04, output);
#ifdef GENERATE_BASELINES
    make_baseline(b04, output);
#else
    EXPECT_EQ(compare_baseline(b04, output), true);
#endif

    // Select 3 parts with 4 targets.
    const char *opt5 =
"selections:\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,9,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,0,0]\n"
"     end:   [9,4,0]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,5,0]\n"
"     end:   [9,9,0]\n"
"target: 4";
    options.reset(); options.parse(opt5, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 4);
    std::string b05 = baseline_file(base + "_05");
    save_visit(b05, output);
#ifdef GENERATE_BASELINES
    make_baseline(b05, output);
#else
    EXPECT_EQ(compare_baseline(b05, output), true);
#endif

    // TODO: try opt5 but target 2 to see if we combine down to 2 domains.
}

//-----------------------------------------------------------------------------
void
test_logical_selection_3d(const std::string &topo, const std::string &base)
{
#ifdef USE_ERROR_HANDLER
    conduit::utils::set_error_handler(tmp_err_handler);
#endif

    // Make 10x10x1 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {11,11,4};
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
    // Override with int64 because YAML loses int/uint information.
    conduit::int64 i100 = 100;
    input["state/cycle"].set(i100);
    input["state/domain_id"].set((int)0);

    // With no options (turn mapping off though because otherwise we add 
    // the original vertex and element fields), test that output==input
    const char *opt0 =
"mapping: 0";
    options.reset(); options.parse(opt0, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(input.diff(output, msg, 0.0, true), false) << msg.to_json();
    std::string b00 = baseline_file(base + "_00");
    save_visit(b00, output);

    // Select the whole thing but divide it into target domains.
    const char *opt1 =
"target: 2";
    options.reset(); options.parse(opt1, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 2);
    std::string b01 = baseline_file(base + "_01");
    save_visit(b01, output);
#ifdef GENERATE_BASELINES
    make_baseline(b01, output);
#else
    EXPECT_EQ(compare_baseline(b01, output), true);
#endif

    // Select the whole thing but divide it into target domains.
    const char *opt2 =
"target: 4";
    options.reset(); options.parse(opt2, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 4);
    std::string b02 = baseline_file(base + "_02");
    save_visit(b02, output);
#ifdef GENERATE_BASELINES
    make_baseline(b02, output);
#else
    EXPECT_EQ(compare_baseline(b02, output), true);
#endif

    // Select the whole thing but go out of bounds to see if the selections
    // clamp to good values.
    const char *opt3 =
"selections:\n"
"   -\n"
"     type: \"logical\"\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [100,100,100]";
    options.reset(); options.parse(opt3, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    EXPECT_EQ(conduit::blueprint::mesh::is_multi_domain(output), false);
    std::string b03 = baseline_file(base + "_03");
    save_visit(b03, output);
#ifdef GENERATE_BASELINES
    make_baseline(b03, output);
#else
    EXPECT_EQ(compare_baseline(b03, output), true);
#endif

    // Select 3 parts.
    const char *opt4 =
"selections:\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,9,2]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,0,0]\n"
"     end:   [9,4,2]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,5,0]\n"
"     end:   [9,9,2]";
    options.reset(); options.parse(opt4, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 3);
    std::string b04 = baseline_file(base + "_04");
    save_visit(b04, output);
#ifdef GENERATE_BASELINES
    make_baseline(b04, output);
#else
    EXPECT_EQ(compare_baseline(b04, output), true);
#endif

    // Select 3 parts with 4 targets.
    const char *opt5 =
"selections:\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,9,2]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,0,0]\n"
"     end:   [9,4,2]\n"
"   -\n"
"     type: logical\n"
"     domain_id: 0\n"
"     start: [5,5,0]\n"
"     end:   [9,9,2]\n"
"target: 4";
    options.reset(); options.parse(opt5, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 4);
    std::string b05 = baseline_file(base + "_05");
    save_visit(b05, output);
#ifdef GENERATE_BASELINES
    make_baseline(b05, output);
#else
    EXPECT_EQ(compare_baseline(b05, output), true);
#endif

    // TODO: try opt5 but target 2 to see if we combine down to 2 domains.
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_logical_2d)
{
    test_logical_selection_2d("uniform", "uniform_logical_2d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, rectilinear_logical_2d)
{
    test_logical_selection_2d("rectilinear", "rectilinear_logical_2d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, structured_logical_2d)
{
    test_logical_selection_2d("structured", "structured_logical_2d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_logical_3d)
{
    test_logical_selection_3d("uniform", "uniform_logical_3d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, rectilinear_logical_3d)
{
    test_logical_selection_3d("rectilinear", "rectilinear_logical_3d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, structured_logical_3d)
{
    test_logical_selection_3d("structured", "structured_logical_3d");
}

//-----------------------------------------------------------------------------
void
test_explicit_selection(const std::string &topo, const conduit::index_t vdims[3],
    const std::string &base, int (*spc)(conduit::index_t, conduit::index_t))
{
#ifdef USE_ERROR_HANDLER
    conduit::utils::set_error_handler(tmp_err_handler);
#endif

    // Make 10x10x1 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
    // Override with int64 because YAML loses int/uint information.
    conduit::int64 i100 = 100;
    input["state/cycle"].set(i100);

    conduit::index_t nelem = conduit::blueprint::mesh::topology::length(input["topologies"][0]);

    // Select the whole thing. Check output==input
    options.reset();
    {
        std::vector<conduit::index_t> elem;
        for(conduit::index_t i = 0; i < nelem; i++)
            elem.push_back(i);
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "explicit";
        sel1["elements"] = elem;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    std::string b00 = baseline_file(base + "_00");
    save_visit(b00, output);
#ifdef GENERATE_BASELINES
    make_baseline(b00, output);
#else
    EXPECT_EQ(compare_baseline(b00, output), true);
#endif

    // Select half of the cells.
    options.reset();
    {
        auto n2 = nelem / 2;
        std::vector<conduit::index_t> elem;
        for(conduit::index_t i = 0; i < n2; i++)
            elem.push_back(i);
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "explicit";
        sel1["elements"] = elem;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    std::string b01 = baseline_file(base + "_01");
    save_visit(b01, output);
#ifdef GENERATE_BASELINES
    make_baseline(b01, output);
#else
    EXPECT_EQ(compare_baseline(b01, output), true);
#endif

    // Select a checkerboard of cells
    options.reset();
    std::vector<conduit::index_t> elem;
    {
        conduit::index_t ci = 0;
        for(conduit::index_t j = 0; j < vdims[1]-1; j++)
        for(conduit::index_t i = 0; i < vdims[0]-1; i++)
        {
            int n = spc(i,j);
            for(int k = 0; k < n; k++)
            {
                if((i+j) % 2 == 0)
                    elem.push_back(ci);
                ci++;
            }
        }

        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "explicit";
        sel1["elements"] = elem;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    std::string b02 = baseline_file(base + "_02");
    save_visit(b02, output);
#ifdef GENERATE_BASELINES
    make_baseline(b02, output);
#else
    EXPECT_EQ(compare_baseline(b02, output), true);
#endif

    // Make 2 selections
    options.reset();
    {
        auto n2 = nelem / 2;
        std::vector<conduit::index_t> elem, elem2;
        for(conduit::index_t i = 0; i < nelem; i++)
        {
            if(i < n2)
                elem.push_back(i);
            else
                elem2.push_back(i);
        }
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "explicit";
        sel1["elements"] = elem;
        conduit::Node &sel2 = options["selections"].append();
        sel2["type"] = "explicit";
        sel2["elements"] = elem2;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 2);
    std::string b03 = baseline_file(base + "_03");
    save_visit(b03, output);
#ifdef GENERATE_BASELINES
    make_baseline(b03, output);
#else
    EXPECT_EQ(compare_baseline(b03, output), true);
#endif

    // Take previous 2 domain selection and partition it into 5 domains.
    options["target"] = 5;
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 5);
    std::string b04 = baseline_file(base + "_04");
    save_visit(b04, output);
#ifdef GENERATE_BASELINES
    make_baseline(b04, output);
#else
    EXPECT_EQ(compare_baseline(b04, output), true);
#endif
}

//-----------------------------------------------------------------------------
int spc(conduit::index_t i, conduit::index_t j)
{
    return 1;
}

//-----------------------------------------------------------------------------
int quads_and_tris_spc(conduit::index_t i, conduit::index_t j)
{
    return (i % 2 == 0) ? 1 : 2;
}

//-----------------------------------------------------------------------------
int hexs_and_tets_spc(conduit::index_t i, conduit::index_t j)
{
    int n;
    if((i == 0 && j == 0) || (i == 0 && j == 5))
        n = 1;
    else if(i == 1 && j == 0)
        n = 6;
    else if(j < 5)
        n = 1;
    else
        n = 6;
    return n;
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_explicit_2d)
{
    conduit::index_t vdims[] = {11,11,1};
    test_explicit_selection("uniform", vdims, "uniform_explicit_2d", spc);
}

//-----------------------------------------------------------------------------
// NOTE: VisIt does not support polygonal cells from Blueprint at this time so
//       I could not visually verify the mesh. The files look okay.
TEST(conduit_blueprint_mesh_partition, quads_poly_explicit_2d)
{
    conduit::index_t vdims[] = {11,11,0};
    test_explicit_selection("quads_poly", vdims, "quads_poly_explicit_2d", spc);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, hexs_explicit_2d)
{
    conduit::index_t vdims[] = {11,11,2};
    test_explicit_selection("hexs", vdims, "hexs_explicit_3d", spc);
}

//-----------------------------------------------------------------------------
// NOTE: verified by converting to VTK in another tool.
TEST(conduit_blueprint_mesh_partition, hexs_poly_explicit_3d)
{
    conduit::index_t vdims[] = {11,11,2};
    test_explicit_selection("hexs_poly", vdims, "hexs_poly_explicit_3d", spc);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, quads_and_tris_explicit_2d)
{
    conduit::index_t vdims[] = {11,11,0};
    test_explicit_selection("quads_and_tris", vdims, "quads_end_tris_explicit_2d",
        quads_and_tris_spc);
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, hexs_and_tets_explicit_3d)
{
    conduit::index_t vdims[] = {11,11,2};
    test_explicit_selection("hexs_and_tets", vdims, "hexs_and_tets_explicit_3d",
        hexs_and_tets_spc);
}

//-----------------------------------------------------------------------------
void
test_ranges_selection_2d(const std::string &topo, const std::string &base)
{
#ifdef USE_ERROR_HANDLER
    conduit::utils::set_error_handler(tmp_err_handler);
#endif

    // Make 10x10x1 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {11,11,1};
    if(topo == "quads")
    {
        vdims[2] = 0;
    }
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
    // Override with int64 because YAML loses int/uint information.
    conduit::int64 i100 = 100;
    input["state/cycle"].set(i100);

    conduit::index_t nelem = conduit::blueprint::mesh::topology::length(input["topologies"][0]);
    auto n2 = nelem / 2;

    // Select the whole thing. Check output==input
    options.reset();
    {
        std::vector<conduit::index_t> ranges;
        ranges.push_back(0);
        ranges.push_back(nelem-1);
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "ranges";
        sel1["ranges"] = ranges;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    std::string b00 = baseline_file(base + "_00");
    save_visit(b00, output);
#ifdef GENERATE_BASELINES
    make_baseline(b00, output);
#else
    EXPECT_EQ(compare_baseline(b00, output), true);
#endif

    // Select out of bounds to make sure it clamps the range.
    options.reset();
    {
        std::vector<conduit::index_t> ranges;
        ranges.push_back(0);
        ranges.push_back(nelem * 2);
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "ranges";
        sel1["ranges"] = ranges;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    std::string b01 = baseline_file(base + "_01");
    save_visit(b01, output);
#ifdef GENERATE_BASELINES
    make_baseline(b01, output);
#else
    EXPECT_EQ(compare_baseline(b01, output), true);
#endif

    // Select a single cell
    options.reset();
    {
        std::vector<conduit::index_t> ranges;
        ranges.push_back(0);
        ranges.push_back(0);
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "ranges";
        sel1["ranges"] = ranges;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    std::string b02 = baseline_file(base + "_02");
    save_visit(b02, output);
#ifdef GENERATE_BASELINES
    make_baseline(b02, output);
#else
    EXPECT_EQ(compare_baseline(b02, output), true);
#endif

    // Select multiple ranges
    options.reset();
    {
        std::vector<conduit::index_t> ranges;
        // bottom half
        ranges.push_back(0);
        ranges.push_back(n2-1);
        // top row
        ranges.push_back((vdims[1]-1-1) * (vdims[0]-1));
        ranges.push_back((vdims[1]-1)   * (vdims[0]-1));
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "ranges";
        sel1["ranges"] = ranges;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 1);
    std::string b03 = baseline_file(base + "_03");
    save_visit(b03, output);
#ifdef GENERATE_BASELINES
    make_baseline(b03, output);
#else
    EXPECT_EQ(compare_baseline(b03, output), true);
#endif

    // Multiple range selections
    options.reset();
    {
        std::vector<conduit::index_t> ranges, ranges2;
        // domain 1
        ranges.push_back(0);
        ranges.push_back(n2-1);
        // domain 2
        ranges2.push_back(n2);
        ranges2.push_back(nelem-1);
        conduit::Node &sel1 = options["selections"].append();
        sel1["type"] = "ranges";
        sel1["ranges"] = ranges;
        conduit::Node &sel2 = options["selections"].append();
        sel2["type"] = "ranges";
        sel2["ranges"] = ranges2;
    }
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 2);
    std::string b04 = baseline_file(base + "_04");
    save_visit(b04, output);
#ifdef GENERATE_BASELINES
    make_baseline(b04, output);
#else
    EXPECT_EQ(compare_baseline(b04, output), true);
#endif

    // Take previous selection and target 5 domains to test partitioning
    options["target"] = 5;
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 5);
    std::string b05 = baseline_file(base + "_05");
    save_visit(b05, output);
#ifdef GENERATE_BASELINES
    make_baseline(b05, output);
#else
    EXPECT_EQ(compare_baseline(b05, output), true);
#endif
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, uniform_ranges_2d)
{
    test_ranges_selection_2d("uniform", "uniform_ranges_2d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, rectilinear_ranges_2d)
{
    test_ranges_selection_2d("rectilinear", "rectilinear_ranges_2d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, structured_ranges_2d)
{
    test_ranges_selection_2d("structured", "structured_ranges_2d");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, quads_ranges_2d)
{
    test_ranges_selection_2d("quads", "quads_ranges_2d");
}

// TODO: Figure out where the point_merge class can live so that
//  these tests can be re-enabled.
#if DO_POINT_MERGE_TESTS
//-----------------------------------------------------------------------------
//-- Point merge
//-----------------------------------------------------------------------------
// The tolerance used by all of the point merge tests
static const double tolerance = 0.00001;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition_point_merge, one)
{
    conduit::Node braid;
    conduit::blueprint::mesh::examples::braid("tets", 2, 2, 2, braid);
    auto &braid_coordset = braid["coordsets/coords"];

    // Test that 1 coordset in gives the exact same coordset out
    std::vector<const conduit::Node*> one;
    one.push_back(&braid_coordset);

    conduit::Node opts;
    opts["merge_tolerance"] = tolerance;

    conduit::Node output;
    conduit::blueprint::mesh::coordset::combine(one, output, &opts);

    conduit::Node info;
    bool different = braid_coordset.diff(output["coordsets/coords"], info);
    EXPECT_FALSE(different);
    if(different || always_print)
    {
        braid_coordset.print();
        output.print();
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition_point_merge, same)
{
    // Test that two identical coordsets create the same coordset
    conduit::Node braid;
    conduit::blueprint::mesh::examples::braid("tets", 2, 2, 2, braid);
    auto &braid_coordset = braid["coordsets/coords"];

    std::vector<const conduit::Node*> same;
    same.push_back(&braid_coordset); same.push_back(&braid_coordset);

    conduit::Node opts;
    opts["merge_tolerance"] = tolerance;

    conduit::Node output;
    conduit::blueprint::mesh::coordset::combine(same, output, &opts);

    conduit::Node info;
    bool different = braid_coordset["type"].diff(output["type"], info);
    different &= braid_coordset["values"].diff(output["values"], info);
    EXPECT_FALSE(different);
    if(different || always_print)
    {
        std::cout << "Input (x2):" << std::endl;
        braid_coordset.print();
        std::cout << "Output:" << std::endl;
        output.print();
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition_point_merge, different)
{
    // Test that two different coordsets create the union of their coordinates
    conduit::Node braid;
    conduit::blueprint::mesh::examples::braid("tets", 2, 2, 2, braid);
    auto &braid_coordset = braid["coordsets/coords"];
    conduit::Node polytess;
    conduit::blueprint::mesh::examples::polytess(1, 1, polytess);
    auto &polytess_coordset = polytess["coordsets/coords"];

    std::vector<const conduit::Node*> different;
    different.push_back(&braid_coordset); different.push_back(&polytess_coordset);

    conduit::Node opts;
    opts["merge_tolerance"] = tolerance;

    conduit::Node output;
    conduit::blueprint::mesh::coordset::combine(different, output, &opts);

    conduit::Node info;
    bool is_different0 = different[0]->diff(output["coordsets/coords"], info);
    EXPECT_TRUE(is_different0);
    bool is_different1 = different[1]->diff(output["coordsets/coords"], info);
    EXPECT_TRUE(is_different1);

    static const std::string filename = baseline_file("pointmerge_different");
#ifdef GENERATE_BASELINES
    make_baseline(filename, output);
#else
    conduit::Node ans; load_baseline(filename, ans);
    // NOTE: If rebaselined we won't need to compare paths anymore
    bool is_output_different = ans["coordsets/coords/type"].diff(output["type"], info);
    is_output_different |= ans["coordsets/coords/values"].diff(output["values"], info);
    is_output_different |= ans["pointmaps"].diff(output["pointmaps"], info);
    EXPECT_FALSE(is_output_different);
    if(is_output_different)
    {
        std::cout << "Baseline:" << std::endl;
        ans.print();
        std::cout << "Output:" << std::endl;
        output.print();
    }
#endif
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition_point_merge, multidomain4)
{
    // Use the spiral example to attach domains that we know are connected
    conduit::Node spiral;
    conduit::blueprint::mesh::examples::spiral(4, spiral);
    std::vector<const conduit::Node*> multidomain;

    const conduit::index_t ndom = spiral.number_of_children();
    for(conduit::index_t i = 0; i < ndom; i++)
    {
        conduit::Node &dom = spiral.child(i);
        multidomain.push_back(dom.fetch_ptr("coordsets/coords"));
    }

    conduit::Node opts;
    opts["merge_tolerance"] = tolerance;

    conduit::Node output;
    conduit::blueprint::mesh::coordset::combine(multidomain, output, &opts);

    static const std::string filename = baseline_file("pointmerge_multidomain4");
#ifdef GENERATE_BASELINES
    make_baseline(filename, output);
#else
    conduit::Node ans; load_baseline(filename, ans);
    conduit::Node info;
    // NOTE: If rebaselined we won't need to compare paths anymore
    bool is_different = ans["coordsets/coords/type"].diff(output["type"], info);
    is_different |= ans["coordsets/coords/values"].diff(output["values"], info);
    is_different |= ans["pointmaps"].diff(output["pointmaps"], info);
    EXPECT_FALSE(is_different);
    if(is_different || always_print)
    {
        std::cout << "Baseline:" << std::endl;
        ans.print();
        std::cout << "Output:" << std::endl;
        output.print();
    }
#endif
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition_point_merge, multidomain8)
{
    // Use the spiral example to attach domains that we know are connected, with more data
    //   this makes sure that multiple nodes are used in the spatial search implementation
    conduit::Node spiral;
    conduit::blueprint::mesh::examples::spiral(8, spiral);
    std::vector<const conduit::Node*> multidomain;

    const conduit::index_t ndom = spiral.number_of_children();
    for(conduit::index_t i = 0; i < ndom; i++)
    {
        conduit::Node &dom = spiral.child(i);
        multidomain.push_back(dom.fetch_ptr("coordsets/coords"));
    }

    conduit::Node opts;
    opts["merge_tolerance"] = tolerance;

    conduit::Node output;
    conduit::blueprint::mesh::coordset::combine(multidomain, output, &opts);

    static const std::string filename = baseline_file("pointmerge_multidomain8");
#ifdef GENERATE_BASELINES
    make_baseline(filename, output);
#else
    conduit::Node ans; load_baseline(filename, ans);
    conduit::Node info;
    // NOTE: If rebaselined we won't need to compare paths anymore
    bool is_different = ans["coordsets/coords/type"].diff(output["type"], info);
    is_different |= ans["coordsets/coords/values"].diff(output["values"], info);
    is_different |= ans["pointmaps"].diff(output["pointmaps"], info);
    EXPECT_FALSE(is_different);
    if(is_different || always_print)
    {
        std::cout << "Baseline:" << std::endl;
        ans.print();
        std::cout << "Output:" << std::endl;
        output.print();
    }
#endif
}
#endif

//-----------------------------------------------------------------------------
//-- Combine topology --
//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_combine, recombine_braid)
{
    const auto recombine_braid_case = [](const std::string &case_name, const conduit::index_t *vdims)
    {
        std::cout << "-------- Start case " << case_name << " --------" << std::endl;
        conduit::Node braid;
        conduit::blueprint::mesh::examples::braid(case_name, vdims[0], vdims[1], vdims[2], braid);
    #ifdef DEBUG_RECOMBINE_BRAID
        const std::string base_name = "recombine_" + case_name;
        save_visit(base_name + "_original", braid);
    #endif

        // First split the braid mesh
        conduit::Node split;
        {
            static const std::string split_yaml = "target: 2";

            conduit::Node split_opts; split_opts.parse(split_yaml, "yaml");

            conduit::blueprint::mesh::partition(braid, split_opts, split);
        #ifdef DEBUG_RECOMBINE_BRAID
            save_visit(base_name + "_split", split);
        #endif

            conduit::Node verify_info;
            bool is_valid = conduit::blueprint::mesh::verify(split, verify_info);
            if(!is_valid)
            {
                verify_info.print();
            }
            EXPECT_TRUE(is_valid);
        }

        // Now put it back together
        conduit::Node combine;
        {
            static const std::string combine_yaml = R"(target: 1)";
            conduit::Node combine_opts; combine_opts.parse(combine_yaml, "yaml");

            std::vector<const conduit::Node*> chunks;
            std::vector<conduit_index_t> chunk_ids;
            for(conduit_index_t i = 0; i < split.number_of_children(); i++)
            {
                chunks.push_back(&split[i]);
                chunk_ids.push_back(i);
            }

            conduit::blueprint::mesh::Partitioner p;
            p.combine(0, chunks, chunk_ids, combine);
        #ifdef DEBUG_RECOMBINE_BRAID
            save_visit(base_name + "_combined", combine);
        #endif
            conduit::Node verify_info;
            bool is_valid = conduit::blueprint::mesh::verify(combine, verify_info);
            if(!is_valid)
            {
                verify_info.print();
            }
            EXPECT_TRUE(is_valid);
        }

        // Compare combined mesh to baselines
        const std::string filename = baseline_file("recombine_braid_" + case_name);
    #ifdef GENERATE_BASELINES
        make_baseline(filename, combine);
    #else
        conduit::Node ans; load_baseline(filename, ans);
        conduit::Node info;
        bool is_different = ans.diff(combine, info, CONDUIT_EPSILON, true);
        EXPECT_FALSE(is_different);
        if(is_different || always_print)
        {
            info.print();
        }
    #endif

        std::cout << "-------- End case " << case_name << "   --------" << std::endl;
    };

    static const conduit::index_t dims2[] = {11,11,0};
    static const std::array<std::string, 4> cases2 = {
        "tris",
        "quads",
        "quads_poly",
        "quads_and_tris"
    //    "quads_and_tris_offsets"
    };
    for(const auto &c : cases2)
    {
        recombine_braid_case(c, dims2);
    }

    static const conduit::index_t dims3[] = {3,3,2};
    static const std::array<std::string, 4> cases3 = {
        "tets",
        "hexs",
        "hexs_poly",
        "hexs_and_tets"
    };
    for(const auto &c : cases3)
    {
        recombine_braid_case(c, dims3);
    }
}

//-----------------------------------------------------------------------------
// #define DEBUG_COMBINE_MULTIDOMAIN
TEST(conduit_blueprint_mesh_combine, multidomain)
{
    const std::string base_name = "combine_multidomain";

    const auto combine_multidomain_case = [&base_name](conduit::index_t ndom) {
        std::cout << "-------- Start case " << ndom << " --------" << std::endl;
        conduit::Node spiral;
        conduit::blueprint::mesh::examples::spiral(ndom, spiral);
    #ifdef DEBUG_COMBINE_MULTIDOMAIN
        save_visit(base_name + std::to_string(ndom) + "_input", spiral);
    #endif

        static const std::string opts_yaml = "target: 1";
        conduit::Node opts; opts.parse(opts_yaml, "yaml");

        conduit::Node output;
        conduit::blueprint::mesh::partition(spiral, opts, output);

    #ifdef DEBUG_COMBINE_MULTIDOMAIN
        save_visit(base_name + std::to_string(ndom) + "_output", output);
    #endif

    // TODO: Rebaseline when rectilinear -> rectilinear is supported
    const std::string filename = baseline_file(base_name + std::to_string(ndom));
    #ifdef GENERATE_BASELINES
        make_baseline(filename, output);
    #else
        conduit::Node ans; load_baseline(filename, ans);
        conduit::Node info;
        bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
        EXPECT_FALSE(is_different);
        if(is_different || always_print)
        {
            info.print();
        }
    #endif
        std::cout << "-------- End case " << ndom << " --------" << std::endl;
    };

    std::array<conduit::index_t, 4> cases{
        2,
        4,
        7,
        9
    };
    for(const auto c : cases)
    {
        combine_multidomain_case(c);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_combine, to_poly)
{
    const auto to_polys_case = [](const std::string &case_name, const conduit::index_t vdims[3])
    {
        std::cout << "-------- Start case " << case_name << " --------" << std::endl;
        const std::string base_name = "combine_to_poly_" + case_name;

        // Make a polygonal or polyhedral domain
        conduit::Node poly_braid;
        if(vdims[2] > 1)
        {
            conduit::blueprint::mesh::examples::braid("hexs_poly",
                                                      2,
                                                      2,
                                                      2,
                                                      poly_braid);

            // Move the points to the side
            std::array<std::array<double, 8>, 3> new_coords = {{
                {-30.0, -10.0, -30.0, -10.0, -30.0, -10.0, -30.0, -10.0},
                {-30.0, -30.0, -10.0, -10.0, -30.0, -30.0, -10.0, -10.0},
                {-30.0, -30.0, -30.0, -30.0, -10.0, -10.0, -10.0, -10.0}
            }};
            conduit::Node &n_coords = poly_braid["coordsets/coords/values"];
            for(conduit::index_t d = 0; d < 3; d++)
            {
                conduit::Node &n_dim = n_coords[d];
                if(n_dim.dtype().is_float32())
                {
                    conduit::float32_array vals = n_dim.value();
                    for(conduit::index_t vi = 0; vi < 8; vi++)
                    {
                        vals[vi] = new_coords[d][vi];
                    }
                }
                else if(n_dim.dtype().is_float64())
                {
                    conduit::float64_array vals = n_dim.value();
                    for(conduit::index_t vi = 0; vi < 8; vi++)
                    {
                        vals[vi] = new_coords[d][vi];
                    }
                }
                else
                {
                    CONDUIT_ERROR("Could not translate coordinates from type "
                        << n_dim.dtype().name() << ".");
                }
            }
        }
        else
        {
            conduit::blueprint::mesh::examples::braid("quads_poly", 2,
                2, 0, poly_braid);

            // Move the points to the side
            std::array<std::array<double, 4>, 2> new_coords = {{
                {-30.0, -10.0, -30.0, -10.0},
                {-30.0, -30.0, -10.0, -10.0},
            }};
            conduit::Node &n_coords = poly_braid["coordsets/coords/values"];
            for(conduit::index_t d = 0; d < 2; d++)
            {
                conduit::Node &n_dim = n_coords[d];
                if(n_dim.dtype().is_float32())
                {
                    conduit::float32_array vals = n_dim.value();
                    for(conduit::index_t vi = 0; vi < 4; vi++)
                    {
                        vals[vi] = new_coords[d][vi];
                    }
                }
                else if(n_dim.dtype().is_float64())
                {
                    conduit::float64_array vals = n_dim.value();
                    for(conduit::index_t vi = 0; vi < 4; vi++)
                    {
                        vals[vi] = new_coords[d][vi];
                    }
                }
                else
                {
                    CONDUIT_ERROR("Could not translate coordinates from type "
                        << n_dim.dtype().name() << ".");
                }
            }
        }

        conduit::Node braid;
        conduit::blueprint::mesh::examples::braid(case_name, vdims[0], vdims[1], vdims[2], braid);

        conduit::Node input;
        input["domain_00000"] = poly_braid;
        input["domain_00000/state/domain_id"] = 0;
        input["domain_00001"] = braid;
        input["domain_00001/state/domain_id"] = 1;
    #ifdef DEBUG_TO_POLY
        save_visit(base_name + "_input", input);
    #endif

        const std::string opts_yaml = "target: 1";
        conduit::Node opts; opts.parse(opts_yaml, "yaml");

        conduit::Node output;
        conduit::blueprint::mesh::partition(input, opts, output);

    #ifdef DEBUG_TO_POLY
        save_visit(base_name + "_output", output);
    #endif

        const std::string filename = baseline_file(base_name);
    #ifdef GENERATE_BASELINES
        make_baseline(filename, output);
    #else
        conduit::Node ans; load_baseline(filename, ans);
        conduit::Node info;
        bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
        EXPECT_FALSE(is_different);
        if(is_different || always_print)
        {
            info.print();
        }
    #endif
        std::cout << "-------- End case " << case_name << "   --------" << std::endl;
    };

    static const conduit::index_t dims2[] = {11,11,0};
    static const std::array<std::string, 3> cases2 = {
        "tris",
        "quads",
        "quads_and_tris",
    //    "quads_and_tris_offsets"
    };
    for(const auto &c : cases2)
    {
        to_polys_case(c, dims2);
    }

    static const conduit::index_t dims3[] = {3,3,2};
    static const std::array<std::string, 3> cases3 = {
        "tets",
        "hexs",
        "hexs_and_tets",
    };
    for(const auto &c : cases3)
    {
        to_polys_case(c, dims3);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_combine, uniform)
{
    using namespace conduit::blueprint::mesh::examples;
    const auto uniform_cases = [](bool is3d)
    {
        std::vector<conduit::Node> domains;
        const conduit::index_t nz = (is3d) ? 3 : 1;
        const std::string case_name = (is3d) ? "3d" : "2d";
        const std::string base_file_name = "combine_uniform_" + case_name;
        std::cout << "-------- Start case " << case_name << " --------" << std::endl;
        
        // 0
        domains.emplace_back();
        basic("uniform", 11, 6, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");

        // 1
        domains.emplace_back();
        basic("uniform", 6, 6, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 0;
        domains.back()["coordsets/coords/origin/y"] = 5;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // 2
        domains.emplace_back();
        basic("uniform", 6, 6, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 5;
        domains.back()["coordsets/coords/origin/y"] = 5;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // 3
        domains.emplace_back();
        basic("uniform", 5, 6, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 0;
        domains.back()["coordsets/coords/origin/y"] = 10;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // 4
        domains.emplace_back();
        basic("uniform", 3, 3, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 4;
        domains.back()["coordsets/coords/origin/y"] = 13;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // 5
        domains.emplace_back();
        basic("uniform", 2, 4, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 4;
        domains.back()["coordsets/coords/origin/y"] = 10;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // 6
        domains.emplace_back();
        basic("uniform", 4, 4, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 5;
        domains.back()["coordsets/coords/origin/y"] = 10;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // 7
        domains.emplace_back();
        basic("uniform", 3, 6, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 8;
        domains.back()["coordsets/coords/origin/y"] = 10;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // 8
        domains.emplace_back();
        basic("uniform", 3, 3, nz, domains.back());
        domains.back().remove("coordsets/coords/origin");
        domains.back().remove("coordsets/coords/spacing");
        domains.back()["coordsets/coords/origin/x"] = 6;
        domains.back()["coordsets/coords/origin/y"] = 13;
        if(is3d)
            domains.back()["coordsets/coords/origin/z"] = 0;

        // Nodes that are reused through each partition call
        conduit::Node opts;
        opts["target"] = 1;
        conduit::Node output;

        // Mesh 0
        conduit::Node mesh0;
        for(conduit::index_t i = 0; i < domains.size(); i++)
        {
            domains[i]["state/domain_id"] = i;
            mesh0[(i < 10) 
                ? ("domain_0000" + std::to_string(i))
                : ("domain_000" + std::to_string(i))] = domains[i];
        }
        save_visit(base_file_name + "_mesh0", mesh0);
        std::cout << "mesh0" << std::endl;
        conduit::blueprint::mesh::partition(mesh0, opts, output);
        save_visit(base_file_name + "_mesh0_output", output);

        {
            const std::string filename = baseline_file(base_file_name + "_mesh0");
        #ifdef GENERATE_BASELINES
            make_baseline(filename, output);
        #else
            conduit::Node ans; load_baseline(filename, ans);
            conduit::Node info;
            bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
            EXPECT_FALSE(is_different);
            if(is_different || always_print)
            {
                info.print();
            }
        #endif
        }

        // Mesh1 missing a section
        conduit::Node mesh1;
        for(conduit::index_t i = 0; i < domains.size(); i++)
        {
            if(i == 6)
            {
                if(!is3d)
                {
                    continue;
                }
                else
                {
                    // For 3d make it so the domain exists just not lined up properly
                    domains[i]["coordsets/coords/origin/z"] = 1;
                }
            }
            mesh1[(i < 10) 
                ? ("domain_0000" + std::to_string(i))
                : ("domain_000" + std::to_string(i))] = domains[i];
        }
        save_visit(base_file_name + "_mesh1", mesh1);

        std::cout << "mesh1" << std::endl;
        conduit::blueprint::mesh::partition(mesh1, opts, output);
        save_visit(base_file_name + "_mesh1_output", output);

        {
            const std::string filename = baseline_file(base_file_name + "_mesh1");
        #ifdef GENERATE_BASELINES
            make_baseline(filename, output);
        #else
            conduit::Node ans; load_baseline(filename, ans);
            conduit::Node info;
            bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
            EXPECT_FALSE(is_different);
            if(is_different || always_print)
            {
                info.print();
            }
        #endif
        }

        // Mesh 2
        conduit::Node mesh2;
        mesh2["domain_00000"] = domains[1];
        mesh2["domain_00001"] = domains[2];
        std::cout << "mesh2" << std::endl;
        save_visit(base_file_name + "_mesh2", mesh2);
        conduit::blueprint::mesh::partition(mesh2, opts, output);
        // output.print();
        save_visit(base_file_name + "_mesh2_output", output);

        {
            const std::string filename = baseline_file(base_file_name + "_mesh2");
        #ifdef GENERATE_BASELINES
            make_baseline(filename, output);
        #else
            conduit::Node ans; load_baseline(filename, ans);
            conduit::Node info;
            bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
            EXPECT_FALSE(is_different);
            if(is_different || always_print)
            {
                info.print();
            }
        #endif
        }

        std::cout << "mesh3" << std::endl;
        // change the spacing for domain00001, should suggest rectilinear
        mesh2["domain_00001/coordsets/coords/spacing/dx"] = 0.5;
        mesh2["domain_00001/coordsets/coords/spacing/dy"] = 1.0;
        if(is3d)
            mesh2["domain_00001/coordsets/coords/spacing/dz"] = 1.0;
        conduit::blueprint::mesh::partition(mesh2, opts, output);
        save_visit(base_file_name + "_mesh3_output", output);

        {
            const std::string filename = baseline_file(base_file_name + "_mesh3");
        #ifdef GENERATE_BASELINES
            make_baseline(filename, output);
        #else
            conduit::Node ans; load_baseline(filename, ans);
            conduit::Node info;
            bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
            EXPECT_FALSE(is_different);
            if(is_different || always_print)
            {
                info.print();
            }
        #endif
        }

        std::cout << "-------- End case " << case_name << "   --------" << std::endl;
    };

    uniform_cases(false);
    uniform_cases(true);
}

//-----------------------------------------------------------------------------
TEST(blueprint_mesh_combine, rectilinear)
{
    conduit::Node spiral;
    conduit::blueprint::mesh::examples::spiral(5, spiral);

    conduit::Node opts; opts["target"] = 1;
    conduit::Node combined;
    conduit::blueprint::mesh::partition(spiral, opts, combined);
    save_visit("combine_rectilinear_output1", combined);

    {
        const std::string filename = baseline_file("combine_rectilinear");
    #ifdef GENERATE_BASELINES
        make_baseline(filename, combined);
    #else
        conduit::Node ans; load_baseline(filename, ans);
        conduit::Node info;
        bool is_different = ans.diff(combined, info, CONDUIT_EPSILON, true);
        EXPECT_FALSE(is_different);
        if(is_different || always_print)
        {
            info.print();
        }
    #endif
    }
}

//-----------------------------------------------------------------------------
// STRUCTURED COMBINE FUNCTIONS AND TESTS
//-----------------------------------------------------------------------------
void create_structured_domain(conduit::Node &out, const int domain_id,
    const int dimension, const int *dims, const int *reorder,
    const double *origin, const double *scale, const double *g_origin)
{
    using namespace conduit;

    // Reorder the dims
    int real_dims[3];
    for(int i = 0; i < dimension; i++)
    {
        real_dims[i] = dims[reorder[i]];
    }

    // Number of verticies/elements
    int nverts = 1, nelems = 1;
    for(int i = 0; i < dimension; i++)
    {
        nverts *= real_dims[i];
        nelems *= (real_dims[i] - 1);
    }


    // Define the mesh in conduit blueprint form
    out["state/domain_id"] = (index_t)domain_id;
    out["topologies/mesh/type"] = "structured";
    out["topologies/mesh/coordset"] = "coords";
    out["topologies/mesh/elements/dims/i"] = real_dims[0]-1;
    out["topologies/mesh/elements/dims/j"] = real_dims[1]-1;
    if(dimension > 2)
        out["topologies/mesh/elements/dims/k"] = real_dims[2]-1;
    out["coordsets/coords/type"] = "explicit";
    Schema s;
    s["x"].set(DataType::c_double(nverts,               0, sizeof(double)*dimension));
    s["y"].set(DataType::c_double(nverts,   sizeof(double), sizeof(double)*dimension));
    if(dimension > 2)
        s["z"].set(DataType::c_double(nverts, 2*sizeof(double), sizeof(double)*dimension));
    out["coordsets/coords/values"].set(s);
    out["fields/vert_field/topology"] = "mesh";
    out["fields/vert_field/association"] = "vertex";
    out["fields/vert_field/values"].set(DataType::c_double(nverts));
    out["fields/dist/topology"] = "mesh";
    out["fields/dist/association"] = "vertex";
    out["fields/dist/values"].set(DataType::c_double(nverts));
    out["fields/elem_field/topology"] = "mesh";
    out["fields/elem_field/association"] = "element";
    out["fields/elem_field/values"].set(DataType::c_double(nelems));

    // Extract the pointers to the allocated memory
    double *coords = (double*)out["coordsets/coords/values"].element_ptr(0);
    double *vfield = (double*)out["fields/vert_field/values"].element_ptr(0);
    double *dist   = (double*)out["fields/dist/values"].element_ptr(0);
    double *efield = (double*)out["fields/elem_field/values"].element_ptr(0);

    int idx = 0;
    int id  = 0;
    if(dimension == 3)
    {
        for(int k = 0; k < real_dims[2]; k++)
        {
            for(int j = 0; j < real_dims[1]; j++)
            {
                for(int i = 0; i < real_dims[0]; i++, id++, idx+=3)
                {
                    const double temp[3] = {
                        (double)i * scale[reorder[0]],
                        (double)j * scale[reorder[1]],
                        (double)k * scale[reorder[2]]
                    };
                    coords[idx]   = origin[0] + temp[reorder[0]];
                    coords[idx+1] = origin[1] + temp[reorder[1]];
                    coords[idx+2] = origin[2] + temp[reorder[2]];
                    vfield[id] = id;
                    
                    const double dx = coords[idx]   - g_origin[0];
                    const double dy = coords[idx+1] - g_origin[1];
                    const double dz = coords[idx+2] - g_origin[2];
                    dist[id] = std::sqrt(dx*dx + dy*dy + dz*dz);
                }
            }
        }

        id = 0;
        for(int k = 0; k < real_dims[2]-1; k++)
        {
            for(int j = 0; j < real_dims[1]-1; j++)
            {
                for(int i = 0; i < real_dims[0]-1; i++, id++)
                {
                    efield[id] = id;
                }
            }
        }
    }
    else if(dimension == 2)
    {
        for(int j = 0; j < real_dims[1]; j++)
        {
            for(int i = 0; i < real_dims[0]; i++, id++, idx+=2)
            {
                const double temp[2] = {(double)i * scale[0], (double)j * scale[1]};
                coords[idx]   = origin[0] + temp[reorder[0]];
                coords[idx+1] = origin[1] + temp[reorder[1]];
                vfield[id] = id;
                
                const double dx = coords[idx]   - g_origin[0];
                const double dy = coords[idx+1] - g_origin[1];
                dist[id] = std::sqrt(dx*dx + dy*dy);
            }
        }

        id = 0;
        for(int j = 0; j < real_dims[1]-1; j++)
        {
            for(int i = 0; i < real_dims[0]-1; i++, id++)
            {
                efield[id] = id;
            }
        }
    }
}

//-----------------------------------------------------------------------------
void create_grain_case(conduit::Node &out)
{
    out.reset();

    // For all domains
    const int dims[3] = {3, 4, 2};
    const double g_origin[3] = {0., 0., 0.};
    
    // Cases
    const int origin_x[8] = {0,    1,   0,    1,   0,   1,     0,   1};
    const int origin_y[8] = {0,    0,   1,    1,   0,   0,     1,   1};
    const double scale_x[8]= {1., -1.,   1.,  -1.,  1., -1.,   1.,  -1.};
    const double scale_y[8]= {1.,  1.,  -1.,  -1.,  1.,  1.,  -1.,  -1.};
    const int reorder_x[8]= {0,    0,    0,    0,   1,   1,     1,   1};
    const int reorder_y[8]= {1,    1,    1,    1,   0,   0,     0,   0};
    int id = 0;

    // TODO: Make a 2D case
    // TODO: Also K
    for(int j = 0; j < 8; j++)
    {
        // {ymin, ymax}
        int y[2];
        y[0] = (int)g_origin[1] + (j * (dims[1]-1));
        y[1] = (int)g_origin[1] + ((j+1) * (dims[1]-1));

        const int start_case = j;
        for(int i = 0; i < 8; i++, id++)
        {
            // {xmin, xmax}
            int x[2];
            x[0] = (int)g_origin[0] + (i * (dims[0]-1));
            x[1] = (int)g_origin[0] + (i+1) * (dims[0]-1);

            const int c = (start_case + i) % 8;
            double o[3];
            double s[3];
            int   r[3];
            o[0] = (double)x[origin_x[c]];
            o[1] = (double)y[origin_y[c]];
            o[2] = 0;
            s[0] = scale_x[c];
            s[1] = scale_y[c];
            s[2] = 1.f;
            r[0] = reorder_x[c];
            r[1] = reorder_y[c];
            r[2] = 2;

            std::string domain_name = (id > 10 ? "domain_000" + std::to_string(id) : "domain_0000" + std::to_string(id));
            create_structured_domain(out[domain_name], id, 3, dims, r, o, s, g_origin);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(blueprint_mesh_combine, structured)
{
    const auto braid_cases = [](bool is3d) {
        const std::string base_name = "combine_structured";
        const std::string case_name = (is3d) ? "3d" : "2d";
        const std::string file_name = base_name + "_" + case_name;
        conduit::Node braid;
        conduit::blueprint::mesh::examples::braid("structured", 11, 11, 
            (is3d) ? 11 : 1, braid);
        save_visit("Braid" + case_name + "Structured", braid);

        conduit::Node opts;
        opts["target"] = 2;

        conduit::Node split;
        conduit::blueprint::mesh::partition(braid, opts, split);
        save_visit(file_name + "_input", split);

        opts["target"] = 1;
        conduit::Node combined;
        conduit::blueprint::mesh::partition(split, opts, combined);
        save_visit(file_name + "_output", combined);
        {
            const std::string filename = baseline_file(file_name);
        #ifdef GENERATE_BASELINES
            make_baseline(filename, combined);
        #else
            conduit::Node ans; load_baseline(filename, ans);
            conduit::Node info;
            bool is_different = ans.diff(combined, info, CONDUIT_EPSILON, true);
            EXPECT_FALSE(is_different);
            if(is_different || always_print)
            {
                info.print();
            }
        #endif
        }
    };
    braid_cases(false);
    braid_cases(true);

    const auto curve_test = [](const int *dims, conduit::Node &output) {
        using namespace conduit;
        const double PI_VALUE = 3.14159265359;
        const int N = dims[0]*dims[1]*dims[2];
        const int Nelem = (dims[0]-1)*(dims[1]-1)*(dims[2]-1);

        // Domain 0
        Node &n_d0 = output["domain_00000"];
        {
            const double dt = PI_VALUE / (dims[0]-1);
            int id  = 0;
            int idx = 0;
            int k, j, i;

            n_d0["state/domain_id"] = (index_t)0;
            n_d0["topologies/mesh/type"] = "structured";
            n_d0["topologies/mesh/coordset"] = "coords";
            n_d0["topologies/mesh/elements/dims/i"] = dims[0]-1;
            n_d0["topologies/mesh/elements/dims/j"] = dims[1]-1;
            if(dims[2] > 1)
                n_d0["topologies/mesh/elements/dims/k"] = dims[2]-1;
            n_d0["coordsets/coords/type"] = "explicit";
            Schema s;
            s["x"].set(DataType::c_double(N,               0, sizeof(double)*3));
            s["y"].set(DataType::c_double(N,   sizeof(double), sizeof(double)*3));
            s["z"].set(DataType::c_double(N, 2*sizeof(double), sizeof(double)*3));
            n_d0["coordsets/coords/values"].set(s);
            n_d0["fields/vert_field/topology"] = "mesh";
            n_d0["fields/vert_field/association"] = "vertex";
            n_d0["fields/vert_field/values"].set(DataType::c_double(N));
            n_d0["fields/elem_field/topology"] = "mesh";
            n_d0["fields/elem_field/association"] = "element";
            n_d0["fields/elem_field/values"].set(
                DataType::c_double(Nelem));

            double *coords = (double*)n_d0["coordsets/coords/values"].element_ptr(0);
            double *vfield = (double*)n_d0["fields/vert_field/values"].element_ptr(0);
            double *efield = (double*)n_d0["fields/elem_field/values"].element_ptr(0);

            for(k = 0; k < dims[2]; k++)
            {
                for(j = 0; j < dims[1]; j++)
                {
                    const int j1 = j+1;
                    double t = 0.;
                    for(i = 0; i < dims[0]; i++, id++, t+=dt)
                    {
                        vfield[id]     = id;
#if 1
                        coords[idx++] = std::cos(t) * j1;
                        coords[idx++] = std::sin(t) * j1;
#else
                        coords[idx++] = j1;
                        coords[idx++] = i;
#endif
                        coords[idx++] = k;
                    }
                }
            }

            for(int i = 0; i < Nelem; i++)
            {
                efield[i] = i;
            }
        }

        Node &n_d1 = output["domain_00001"];
        {
            int id  = 0;
            int idx = 0;
            int k, j, i;

            n_d1["state/domain_id"] = (index_t)1;
            n_d1["topologies/mesh/type"] = "structured";
            n_d1["topologies/mesh/coordset"] = "coords";
            n_d1["topologies/mesh/elements/dims/i"] = dims[0]-1;
            n_d1["topologies/mesh/elements/dims/j"] = dims[1]-1;
            if(dims[2] > 1)
                n_d1["topologies/mesh/elements/dims/k"] = dims[2]-1;
            n_d1["coordsets/coords/type"] = "explicit";
            Schema s;
            s["x"].set(DataType::c_double(N,               0, sizeof(double)*3));
            s["y"].set(DataType::c_double(N,   sizeof(double), sizeof(double)*3));
            s["z"].set(DataType::c_double(N, 2*sizeof(double), sizeof(double)*3));
            n_d1["coordsets/coords/values"].set(s);
            n_d1["fields/vert_field/topology"] = "mesh";
            n_d1["fields/vert_field/association"] = "vertex";
            n_d1["fields/vert_field/values"].set(DataType::c_double(N));
            n_d1["fields/elem_field/topology"] = "mesh";
            n_d1["fields/elem_field/association"] = "element";
            n_d1["fields/elem_field/values"].set(
                DataType::c_double((dims[0]-1)*(dims[1]-1)*(dims[2]-1)));

            double *coords = (double*)n_d1["coordsets/coords/values"].element_ptr(0);
            double *vfield = (double*)n_d1["fields/vert_field/values"].element_ptr(0);
            double *efield = (double*)n_d1["fields/elem_field/values"].element_ptr(0);

            for(k = 0; k < dims[2]; k++)
            {
                for(j = 0; j < dims[1]; j++)
                {
                    const int j1 = j+1;
                    double t = 0.;
                    for(i = 0; i < dims[0]; i++, id++)
                    {
                        vfield[id]     = id;
                        coords[idx++] = j1;
                        coords[idx++] = -i;
                        coords[idx++] = k;
                    }
                }
            }

            for(int i = 0; i < Nelem; i++)
            {
                efield[i] = i;
            }
        }
    };

    // TODO: Create a case that combines along a curved face
    {
        conduit::Node curved;
        const int dims[3] = {7,5,3};
        curve_test(dims, curved);
        save_visit("combine_structured_curved_3d_input", curved);

        conduit::Node opts; opts["target"] = 1;
        conduit::Node output;
        conduit::blueprint::mesh::partition(curved, opts, output);
        save_visit("combine_structured_curved_3d_output", output);

        {
            const std::string filename = baseline_file("combine_structured_curved_3d");
        #ifdef GENERATE_BASELINES
            make_baseline(filename, output);
        #else
            conduit::Node ans; load_baseline(filename, ans);
            conduit::Node info;
            bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
            EXPECT_FALSE(is_different);
            if(is_different || always_print)
            {
                info.print();
            }
        #endif
        }
    }

    // TODO: Update the "grain_case" to permute the domains over K
    // TODO: Create a 2D case
    {
        conduit::Node grain;
        create_grain_case(grain);

        save_visit("combine_structured_grain_3d_input", grain);
        
        conduit::Node opts; opts["target"] = 1;
        conduit::Node output;
        conduit::blueprint::mesh::partition(grain, opts, output);
        save_visit("combine_structured_grain_3d_output", output);
        {
            const std::string filename = baseline_file("combine_structured_grain_3d");
        #ifdef GENERATE_BASELINES
            make_baseline(filename, output);
        #else
            conduit::Node ans; load_baseline(filename, ans);
            conduit::Node info;
            bool is_different = ans.diff(output, info, CONDUIT_EPSILON, true);
            EXPECT_FALSE(is_different);
            if(is_different || always_print)
            {
                info.print();
            }
        #endif
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, field_selection)
{
    std::string base("field_selection");
    conduit::Node input, output, options;
    partition::make_field_selection_example(input, -1);
    save_visit("fs", input);

    const char *opt0 =
"selections:\n"
"   -\n"
"     type: field\n"
"     domain_id: 0\n"
"     field: selection_field\n"
"   -\n"
"     type: field\n"
"     domain_id: 1\n"
"     field: selection_field\n"
"   -\n"
"     type: field\n"
"     domain_id: 2\n"
"     field: selection_field\n"
"   -\n"
"     type: field\n"
"     domain_id: 3\n"
"     field: selection_field\n";
    options.reset(); options.parse(opt0, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 6);
    std::string b00 = baseline_file(base + "_00");
    save_visit(b00, output);
#ifdef GENERATE_BASELINES
    make_baseline(b00, output);
#else
    EXPECT_EQ(compare_baseline(b00, output), true);
#endif

    // Test domain_id: any
    const char *opt1 =
"selections:\n"
"   -\n"
"     type: field\n"
"     domain_id: any\n"
"     field: selection_field\n";
    options.reset(); options.parse(opt1, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 6);
    std::string b01 = baseline_file(base + "_01");
    save_visit(b01, output);
#ifdef GENERATE_BASELINES
    make_baseline(b01, output);
#else
    EXPECT_EQ(compare_baseline(b01, output), true);
#endif

    // Test "target: 10". We can split field selections further as
    // explicit selections.
    const char *opt2 =
"selections:\n"
"   -\n"
"     type: field\n"
"     domain_id: any\n"
"     field: selection_field\n"
"target: 10\n";
    options.reset(); options.parse(opt2, "yaml");
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(conduit::blueprint::mesh::number_of_domains(output), 10);
    std::string b02 = baseline_file(base + "_02");
    save_visit(b02, output);
#ifdef GENERATE_BASELINES
    make_baseline(b02, output);
#else
    EXPECT_EQ(compare_baseline(b02, output), true);
#endif
}
#endif

void
my_save_visit(const std::string &filename, const conduit::Node &n)
{
    // NOTE: My VisIt only wants to read HDF5 root files for some reason.
    bool hdf5_enabled = check_if_hdf5_enabled();
    if(!hdf5_enabled) {
        throw std::runtime_error("No hdf5");
    }

    auto pos = filename.rfind("/");
    std::string fn(filename.substr(pos+1,filename.size()-pos-1));
    pos = fn.rfind(".");
    std::string fn_noext(fn.substr(0, pos));

    // Save all the domains to individual files.
    auto ndoms = conduit::blueprint::mesh::number_of_domains(n);
    if(ndoms < 1)
        return;

    // All domains go into subdirectory
    if(!conduit::utils::is_directory(filename))
    {
        conduit::utils::create_directory(filename);
    }

    char dnum[20];
    if(ndoms == 1)
    {
        sprintf(dnum, "%05d", 0);
        std::stringstream ss;
        ss << filename << conduit::utils::file_path_separator()
            << fn_noext << "." << dnum;

        if(hdf5_enabled)
            conduit::relay::io::save(n, ss.str() + ".hdf5", "hdf5");
        // VisIt won't read it:
        conduit::relay::io::save(n, ss.str() + ".yaml", "yaml");
    }
    else
    {
        for(size_t i = 0; i < ndoms; i++)
        {
            sprintf(dnum, "%05d", static_cast<int>(i));
            std::stringstream ss;
            ss << filename << conduit::utils::file_path_separator()
                << fn_noext << "." << dnum;

            if(hdf5_enabled)
                conduit::relay::io::save(n[i], ss.str() + ".hdf5", "hdf5");
            // VisIt won't read it:
            conduit::relay::io::save(n[i], ss.str() + ".yaml", "yaml");
        }
    }

    // Add index stuff to it so we can plot it in VisIt.
    conduit::Node root;
    if(ndoms == 1)
        conduit::blueprint::mesh::generate_index(n, "", ndoms, root["blueprint_index/mesh"]);
    else
        conduit::blueprint::mesh::generate_index(n[0], "", ndoms, root["blueprint_index/mesh"]);
    root["protocol/name"] = "hdf5";
    root["protocol/version"] = CONDUIT_VERSION;
    root["number_of_files"] = ndoms;
    root["number_of_trees"] = ndoms;
    root["file_pattern"] = filename + conduit::utils::file_path_separator() + (fn_noext + ".%05d.hdf5");
    root["tree_pattern"] = "/";
    
    if(hdf5_enabled)
        conduit::relay::io::save(root, fn_noext + "_hdf5.root", "hdf5");

    // VisIt won't read it: 
    root["file_pattern"] = (fn_noext + ".%05d.yaml");
    conduit::relay::io::save(root, fn_noext + "_yaml.root", "yaml");
}

//-----------------------------------------------------------------------------
// Multi-buffer, element-dominant matset
TEST(conduit_blueprint_mesh_partition, matset_multi_by_element)
{
    /// matset_type options:
    ///   full -> non sparse volume fractions and matset values
    ///   sparse_by_material ->  sparse (material dominant) volume fractions
    ///                          and matset values
    ///   sparse_by_element  ->  sparse (element dominant)
    ///                          volume fractions and matset values
    conduit::Node venn;
    conduit::blueprint::mesh::examples::venn("full", 4, 4, 0.33f, venn);

    my_save_visit("venn_multi_by_element", venn);

    conduit::Node venn_part, opts; opts["target"].set(4);
    conduit::blueprint::mesh::partition(venn, opts, venn_part);

    my_save_visit("venn_multi_by_element_partitioned", venn_part);
}

//-----------------------------------------------------------------------------
// Multi-buffer, material-dominant matset
TEST(conduit_blueprint_mesh_partition, matset_multi_by_material)
{
    /// matset_type options:
    ///   full -> non sparse volume fractions and matset values
    ///   sparse_by_material ->  sparse (material dominant) volume fractions
    ///                          and matset values
    ///   sparse_by_element  ->  sparse (element dominant)
    ///                          volume fractions and matset values
    conduit::Node venn;
    conduit::blueprint::mesh::examples::venn("sparse_by_material", 4, 4, 0.33f, venn);

    my_save_visit("venn_multi_by_material", venn);

    conduit::Node venn_part, opts; opts["target"].set(4);
    conduit::blueprint::mesh::partition(venn, opts, venn_part);

    my_save_visit("venn_multi_by_material_partitioned", venn_part);
}

//-----------------------------------------------------------------------------
// Uni-buffer, element-dominant matset
TEST(conduit_blueprint_mesh_partition, matset_uni_by_element)
{
    /// matset_type options:
    ///   full -> non sparse volume fractions and matset values
    ///   sparse_by_material ->  sparse (material dominant) volume fractions
    ///                          and matset values
    ///   sparse_by_element  ->  sparse (element dominant)
    ///                          volume fractions and matset values
    conduit::Node venn;
    conduit::blueprint::mesh::examples::venn("sparse_by_element", 4, 4, 0.33f, venn);

    my_save_visit("venn_uni_by_element", venn);

    conduit::Node venn_part, opts; opts["target"].set(4);
    conduit::blueprint::mesh::partition(venn, opts, venn_part);

    my_save_visit("venn_uni_by_element_partitioned", venn_part);
}

//-----------------------------------------------------------------------------
// Uni-buffer, material-dominant matset
TEST(conduit_blueprint_mesh_partition, matset_uni_by_material)
{
    const int nx = 4;
    const int ny = 4;
    conduit::Node venn;
    conduit::blueprint::mesh::examples::venn("sparse_by_element", nx, ny, 0.33f, venn);

    // Add an element ids field to reverse the matset
    const int N = conduit::blueprint::mesh::topology::length(venn["topologies"][0]);
    std::vector<int> ids;
    for(int i = 0; i < N; i++)
    {
        ids.push_back(i);
    }
    venn["matsets/matset/element_ids"].set(ids);

    my_save_visit("venn_uni_by_material", venn);

    conduit::Node venn_part, opts; opts["target"].set(4);
    conduit::blueprint::mesh::partition(venn, opts, venn_part);

    my_save_visit("venn_uni_by_material_partitioned", venn_part);
}
