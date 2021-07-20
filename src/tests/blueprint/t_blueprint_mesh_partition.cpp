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
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <math.h>
#include <iostream>
#include "gtest/gtest.h"

using std::cout;
using std::endl;

// Enable this macro to generate baselines.
//#define GENERATE_BASELINES

//-----------------------------------------------------------------------------
#ifdef GENERATE_BASELINES
  #ifdef _WIN32
    #include <direct.h>
    void create_path(const std::string &path) { _mkdir(path.c_str()); }
  #else
    #include <sys/stat.h>
    #include <sys/types.h>
    void create_path(const std::string &path) { mkdir(path.c_str(), S_IRWXU); }
  #endif
#else
  void create_path(const std::string &) {}
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
std::string
baseline_file(const std::string &basename)
{
    std::string path(baseline_dir());
    create_path(path);
    path += (sep + std::string("t_blueprint_mesh_partition"));
    create_path(path);
    path += (sep + basename + ".yaml");
    return path;
}

//-----------------------------------------------------------------------------
void
make_baseline(const std::string &filename, const conduit::Node &n)
{
    conduit::relay::io::save(n, filename, "yaml");
}

//-----------------------------------------------------------------------------
bool
compare_baseline(const std::string &filename, const conduit::Node &n)
{
    const double tolerance = 1.e-6;
    conduit::Node baseline, info;
    conduit::relay::io::load(filename, "yaml", baseline);
    const char *line = "*************************************************************";
#if 0
    cout << line << endl;
    baseline.print();
    cout << line << endl;
    n.print();
    cout << line << endl;
#endif

    // Node::diff returns true if the nodes are different. We want not different.
    bool equal = !baseline.diff(n, info, tolerance);

    if(!equal)
    {
       cout << "Difference!" << endl;
       cout << line << endl;
       info.print();
    }
    return equal;
}

//-----------------------------------------------------------------------------
bool
check_if_hdf5_enabled()
{
    conduit::Node io_protos;
    conduit::relay::io::about(io_protos["io"]);
    return io_protos["io/protocols/hdf5"].as_string() == "enabled";
}

//-----------------------------------------------------------------------------
void
save_visit(const std::string &filename, const conduit::Node &n)
{
    // NOTE: My VisIt only wants to read HDF5 root files for some reason.
    bool hdf5_enabled = check_if_hdf5_enabled();

    auto pos = filename.rfind("/");
    std::string fn(filename.substr(pos+1,filename.size()-pos-1));
    pos = fn.rfind(".");
    std::string fn_noext(fn.substr(0, pos));

    // Save all the domains to individual files.
    auto ndoms = conduit::blueprint::mesh::number_of_domains(n);
    if(ndoms < 1)
        return;
    char dnum[20];
    if(ndoms == 1)
    {
        sprintf(dnum, "%05d", 0);
        std::stringstream ss;
        ss << fn_noext << "." << dnum;

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
            ss << fn_noext << "." << dnum;

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
    root["file_pattern"] = (fn_noext + ".%05d.hdf5");
    root["tree_pattern"] = "/";

    if(hdf5_enabled)
        conduit::relay::io::save(root, fn_noext + "_hdf5.root", "hdf5");

    root["file_pattern"] = (fn_noext + ".%05d.yaml");
    // VisIt won't read it:
    conduit::relay::io::save(root, fn_noext + "_yaml.root", "yaml");
}

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
    conduit::utils::set_error_handler(tmp_err_handler);

    // Make 10x10x1 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {11,11,1};
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
    // Override with int64 because YAML loses int/uint information.
    conduit::int64 i100 = 100;
    input["state/cycle"].set(i100);

    // With no options, test that output==input
    conduit::blueprint::mesh::partition(input, options, output);
    EXPECT_EQ(input.diff(output, msg, 0.0), false);
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
"     domain: 0\n"
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
"     domain: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,9,0]\n"
"   -\n"
"     type: logical\n"
"     domain: 0\n"
"     start: [5,0,0]\n"
"     end:   [9,4,0]\n"
"   -\n"
"     type: logical\n"
"     domain: 0\n"
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
"     domain: 0\n"
"     start: [0,0,0]\n"
"     end:   [4,9,0]\n"
"   -\n"
"     type: logical\n"
"     domain: 0\n"
"     start: [5,0,0]\n"
"     end:   [9,4,0]\n"
"   -\n"
"     type: logical\n"
"     domain: 0\n"
"     start: [5,5,0]\n"
"     end:   [9,9,0]\n"
"target: 4";
    options.reset(); options.parse(opt5, "yaml"); options.print();
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
test_logical_selection_3d(const std::string &topo)
{
    conduit::utils::set_error_handler(tmp_err_handler);


    // Make 10x10x10 cell mesh.
    conduit::Node input, output, options, msg;
    conduit::index_t vdims[] = {11,11,11};
    conduit::blueprint::mesh::examples::braid(topo, vdims[0], vdims[1], vdims[2], input);
    conduit::int64 i100 = 100;
    input["state/cycle"].set(i100); // override with int64.

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

#if 0
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
#endif
//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_partition, structured_logical_2d)
{
    test_logical_selection_2d("structured", "structured_logical_2d");
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
