// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_topology_metadata.cpp
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
#include <array>
#include <cmath>
#include "gtest/gtest.h"

#include "blueprint_test_helpers.hpp"

using std::cout;
using std::endl;

using TopologyMetadata = conduit::blueprint::mesh::utils::TopologyMetadata;
using index_t = conduit::index_t;

// Enable this macro to generate baselines.
#define GENERATE_BASELINES

#define USE_ERROR_HANDLER

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
std::string test_name() { return std::string("t_blueprint_mesh_topology_metadata"); }

//-----------------------------------------------------------------------------
int get_rank() { return 0; }

//-----------------------------------------------------------------------------
void barrier() { }

//-----------------------------------------------------------------------------
// Include some helper function definitions
#include "blueprint_baseline_helpers.hpp"

//---------------------------------------------------------------------------
void
yaml_print(std::ostream &os, const conduit::Node &node)
{
    // Override these parameters so we 
    conduit::Node opts;
    opts["num_elements_threshold"] = 10000;
    opts["num_children_threshold"] = 10000;

    std::string s;
    node.to_summary_string_stream(cout, opts);
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
test_topmd(const std::string &base, conduit::Node &topo, conduit::Node &coords)
{
    TopologyMetadata md(topo, coords);
    index_t maxdim = conduit::blueprint::mesh::utils::topology::dims(topo);

    // This node will hold a representation of the metadata.
    conduit::Node rep;

    // Compute topology lengths.
    index_t lengths[4] = {0,0,0,0};
    for(index_t d = maxdim; d >= 0; d--)
    {
        lengths[d] = conduit::blueprint::mesh::utils::topology::length(md.dim_topos[d]);
    }

    // Add all topos to a vis node.
    conduit::Node vis;
    vis["coordsets/coords"].set_external(coords);
    for(index_t d = maxdim; d >= 0; d--)
    {
        // Add the topology to the rep.
        std::stringstream oss;
        oss << "topo" << d;
        std::string tname(oss.str());
        rep[tname].set_external(md.dim_topos[d]);

        // Add the topology to the vis.
        vis[std::string("topologies/") + tname].set_external(md.dim_topos[d]);
    }
    // Save all topos together to a dataset for visualization.
    save_visit(base, vis);

    // Get all the maps and add them to the rep.
    for(int e = maxdim; e >= 0; e--)
    for(int a = maxdim; a >= 0; a--)
    {
        {
            std::stringstream oss;
            oss << "global/map" << e << a;
            std::string mname(oss.str());
            md.get_dim_map(TopologyMetadata::GLOBAL, e, a, rep[mname]);
        }
        {
            std::stringstream oss;
            oss << "local/map" << e << a;
            std::string mname(oss.str());
            md.get_dim_map(TopologyMetadata::LOCAL, e, a, rep[mname]);
        }
    }

#ifdef GENERATE_BASELINES
    std::string b = baseline_file(base);
    make_baseline(b, rep);
#else
    EXPECT_EQ(compare_baseline(b, rep), true);
#endif
}

//-----------------------------------------------------------------------------
void
make_custom_tets(conduit::Node &node)
{
    std::vector<float> xc{0.f, 0.f, 1.f, 0.f, 0.f};
    std::vector<float> yc{0.f, 0.f, 0.f, 1.f, 0.f};
    std::vector<float> zc{0.f, 1.f, 0.f, 0.f, -1.f};
    // 2 tets where they share a face but it is rotated relative to the 2nd tet.
    std::vector<int> conn{0,1,2,3,
                          3,2,0,4
                         };

    node["coordsets/coords/type"] = "explicit";
    node["coordsets/coords/values/x"].set(xc);
    node["coordsets/coords/values/y"].set(yc);
    node["coordsets/coords/values/z"].set(zc);

    node["topologies/mesh/coordset"] = "coords";
    node["topologies/mesh/type"] = "unstructured";
    node["topologies/mesh/elements/shape"] = "tet";
    node["topologies/mesh/elements/connectivity"].set(conn);
}

//-----------------------------------------------------------------------------
void
make_custom_ph(conduit::Node &node)
{
    std::vector<float> xc{0.f, 0.5f, 1.f,
                          0.f, 0.5f, 1.f,
                          0.f, 0.5f, 1.f,
                          0.f, 0.25f, 0.5f, 0.75f, 1.f,
                          0.f, 0.25f, 0.5f, 0.75f, 1.f,
                          0.f, 0.25f, 0.5f, 0.75f, 1.f,
                          0.f, 0.25f, 0.5f, 0.75f, 1.f,
                          0.f, 0.25f, 0.5f, 0.75f, 1.f};
    std::vector<float> yc{0.f,   0.0f,  0.f,
                          0.5f,  0.5f,  0.5f,
                          1.f,   1.f,   1.f,
                          0.f,   0.f,   0.f,   0.f,   0.f,
                          0.25f, 0.25f, 0.25f, 0.25f, 0.25f,
                          0.5f,  0.5f,  0.5f,  0.5f,  0.5f,
                          0.75f, 0.75f, 0.75f, 0.75f, 0.75f,
                          1.f,   1.f,   1.f,   1.f,   1.f};
    std::vector<float> zc{0.f,   0.0f,  0.f,
                          0.f,   0.0f,  0.f,
                          0.f,   0.0f,  0.f,
                          1.f, 1.f, 1.f, 1.f, 1.f,
                          1.f, 1.f, 1.f, 1.f, 1.f,
                          1.f, 1.f, 1.f, 1.f, 1.f,
                          1.f, 1.f, 1.f, 1.f, 1.f,
                          1.f, 1.f, 1.f, 1.f, 1.f};

    std::vector<int> conn{
                         0,2,6,8,12,16,17,20,21,
                         2,4,7,9,13,18,19,22,23,
                         1,3,8,10,14,24,25,28,29,
                         3,5,9,11,15,26,27,30,31
                         };
    std::vector<int> connsize{9,9,9,9};

    std::vector<int> faceconn{// YZ faces
                              0,9,14,19,3,
                              3,19,24,29,6,
                              1,11,16,21,4,
                              4,21,26,31,7,
                              2,13,18,23,5,
                              5,23,28,33,8,
                              // XZ faces
                              0,1,11,10,9,
                              1,2,13,12,11,
                              3,4,21,20,19,
                              4,5,23,22,21,
                              6,7,31,30,29,
                              7,8,33,32,31,
                              // XY faces
                              1,0,3,4,
                              2,1,4,5,
                              4,3,6,7,
                              5,4,7,8,
                              9,10,15,14,
                              10,11,16,15,
                              11,12,17,16,
                              12,13,18,17,
                              14,15,20,19,
                              15,16,21,20,
                              16,17,22,21,
                              17,18,23,22,
                              19,20,25,24,
                              20,21,26,25,
                              21,22,27,26,
                              22,23,28,27,
                              24,25,30,29,
                              25,26,31,20,
                              26,27,32,31,
                              27,28,33,32
                             };
    std::vector<int> facesize{// YZ faces
                              5,5,
                              5,5,
                              5,5,
                              // XZ faces
                              5,5,
                              5,5,
                              5,5,
                              // XY faces
                              4,4,4,4,
                              4,4,4,4,
                              4,4,4,4,
                              4,4,4,4,
                             };
    node["coordsets/coords/type"] = "explicit";
    node["coordsets/coords/values/x"].set(xc);
    node["coordsets/coords/values/y"].set(yc);
    node["coordsets/coords/values/z"].set(zc);

    node["topologies/mesh/coordset"] = "coords";
    node["topologies/mesh/type"] = "unstructured";

    node["topologies/mesh/elements/shape"] = "polyhedral";
    node["topologies/mesh/elements/connectivity"].set(conn);
    node["topologies/mesh/elements/sizes"].set(connsize);

    node["topologies/mesh/subelements/shape"] = "polygonal";
    node["topologies/mesh/subelements/connectivity"].set(faceconn);
    node["topologies/mesh/subelements/sizes"].set(facesize);

    conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(
        node["topologies/mesh"], 
        node["topologies/mesh/elements/offsets"],
        node["topologies/mesh/subelements/offsets"]);

//    yaml_print(cout, node);
//    conduit::Node info;
//    cout << "verify=" << conduit::blueprint::mesh::verify(node, info) << endl;
//    info.print();
}

//-----------------------------------------------------------------------------
void
make_dataset(conduit::Node &node, const std::string &type, conduit::DataType &dtype)
{
    if(type == "custom_tets")
        make_custom_tets(node);
    else if(type == "custom_ph")
        make_custom_ph(node);
    else
    {
        int dims[] = {3,3,2};
        if(type == "lines" || type == "tris" || type == "quads")
            dims[2] = 0;
        conduit::blueprint::mesh::examples::braid(type, dims[0],dims[1],dims[2], node);
    }
    conduit::Node &topo = node["topologies/mesh"];

    // Make sure that these items are the requested data type (if they exist).
    std::vector<std::string> copy_keys{"elements/connectivity",
                                       "elements/sizes",
                                       "elements/offsets",
                                       "subelements/connectivity",
                                       "subelements/sizes",
                                       "subelements/offsets"
                                       };
    for(const auto &key : copy_keys)
    {
        if(topo.has_path(key))
        {
cout << "Copying " << key << " as index_t." << endl;
            conduit::Node &src = topo[key];
            conduit::Node dest;
            dest.set(conduit::DataType(dtype.id(), src.dtype().number_of_elements()));
            src.to_data_type(dtype.id(), dest);
            // replace src with the converted type.
            src.set(dest);
        }
    }
}

//-----------------------------------------------------------------------------
std::vector<index_t>
test_dtype_ids()
{
    return std::vector<index_t>{CONDUIT_INT32_ID, CONDUIT_INT64_ID};
}

//-----------------------------------------------------------------------------
void
test_mesh_type(const std::string &type)
{
#ifdef USE_ERROR_HANDLER
    conduit::utils::set_error_handler(tmp_err_handler);
#endif

    // For each data type, make a mesh that has data of those types and then
    // do the topmd test.
    for(auto dtid : test_dtype_ids())
    {
        conduit::Node node;
        conduit::DataType dtype(dtid);
        make_dataset(node, type, dtype);
        // Now do the test.
        std::stringstream oss;
        oss << "topology_metadata_" << type << "_" << dtype.name();
        test_topmd(oss.str(), node["topologies/mesh"], node["coordsets/coords"]);
    }
}
#if 0
//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, lines)
{
    test_mesh_type("lines");
}
#endif
//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, tris)
{
    test_mesh_type("tris");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, quads)
{
    test_mesh_type("quads");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, tets)
{
    test_mesh_type("tets");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, pyramids)
{
    test_mesh_type("pyramids");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, wedges)
{
    test_mesh_type("wedges");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, hexs)
{
    test_mesh_type("hexs");
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, custom_tets)
{
    test_mesh_type("custom_tets");
}
#if 0
//-----------------------------------------------------------------------------
TEST(conduit_blueprint_topology_metadata, custom_ph)
{
    test_mesh_type("custom_ph");
}
#endif
