// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_blueprint_demos.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
// DO WE NEED THIS:
#include "conduit_blueprint_mesh_utils.hpp"
#include "conduit_relay.hpp"
#include "t_conduit_docs_tutorial_helpers.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

/// Helper Functions ///

//-----------------------------------------------------------------------------
bool
check_if_hdf5_enabled()
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    return io_protos["io/protocols/hdf5"].as_string() == "enabled";
}

// NOTE(JRC): This function is necessary since the Blueprint examples produce
// results with arbitrary data types (e.g. int32, uint32) while JSON-read values
// always use the biggest available data types (e.g. int64, float64), so the
// mesh data needs to be expanded so there aren't false positive diff errors.
void make_mesh_json_compatible(const Node &mesh, Node &emesh)
{
    // TODO(JRC): Add support for list types (isn't currently necessary
    // since they don't appear anywhere in a normal Blueprint schema).
    if(mesh.dtype().id() == DataType::OBJECT_ID ||
        mesh.dtype().id() == DataType::LIST_ID)
    {
        conduit::NodeConstIterator child_it = mesh.children();
        while(child_it.has_next())
        {
            const conduit::Node &child = child_it.next();
            const std::string child_name = child_it.name();
            conduit::Node &echild = emesh[child_name];
            make_mesh_json_compatible(child, echild);
        }
    }
    else // is leaf node
    {
        if(mesh.dtype().is_signed_integer())
        {
            mesh.to_int64_array(emesh);
        }
        else if(mesh.dtype().is_unsigned_integer())
        {
            mesh.to_uint64_array(emesh);
        }
        else if(mesh.dtype().is_floating_point())
        {
            mesh.to_float64_array(emesh);
        }
        else
        {
            emesh.set(mesh);
        }
    }
}

void
test_save_mesh_helper(const conduit::Node &dsets,
                      const std::string &base_name)
{
    Node opts;
    opts["file_style"] = "root_only";
    opts["suffix"] = "none";

    std::cout << "saving to " << base_name << "_yaml" << std::endl;

    relay::io::blueprint::save_mesh(dsets, base_name + "_yaml", "yaml", opts);

    if(check_if_hdf5_enabled())
    {
        std::cout << "saving to " << base_name << "_hdf5" << std::endl;

        relay::io::blueprint::save_mesh(dsets, base_name + "_hdf5", "hdf5", opts);
    }
}

void validate_basic_example(const std::string &name,
                            Node &mesh,
                            const std::string &ref_str)
{
    CONDUIT_INFO("Testing Basic Example '" << name << "'");

    Node info;
    EXPECT_TRUE(blueprint::mesh::verify(mesh, info));
    if(info["valid"].as_string() != "true")
    {
        CONDUIT_INFO(info.to_json());
    }

    Node ref_mesh;
    {
        conduit::Generator gen(ref_str, "json", NULL);
        gen.walk(ref_mesh);
    }

    Node compat_mesh;
    make_mesh_json_compatible(mesh, compat_mesh);
    EXPECT_FALSE(compat_mesh.diff(ref_mesh, info));
    if(info["valid"].as_string() != "true")
    {
        CONDUIT_INFO(info.to_json());
    }

    mesh["topologies/mesh_points/type"] = "points";
    mesh["topologies/mesh_points/coordset"] = "coords";

    std::cout << mesh.to_yaml() << std::endl;

    test_save_mesh_helper(mesh, name);
}

/// Test Functions ///

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_transform_side)
{
    BEGIN_EXAMPLE("blueprint_demo_transform_side");
    // some useful (arbitrary) names
    const std::string SIDE_COORDSET_NAME = "scoords";
    const std::string SIDE_TOPOLOGY_NAME = "stopo";

    // create container node and generate simple uniform 2d 'polygon' mesh
    Node mesh;
    conduit::blueprint::mesh::examples::basic("polygons", 3, 2, 0, mesh);
    // Get a reference to the original (first) mesh topology
    Node &grid_topo = mesh["topologies"].child(0);

    // Set up a new "side" mesh
    Node t2s_map, s2t_map;
    Node &side_coords = mesh["coordsets"][SIDE_COORDSET_NAME];
    Node &side_topo = mesh["topologies"][SIDE_TOPOLOGY_NAME];
    conduit::blueprint::mesh::topology::unstructured::
        generate_sides(grid_topo, side_topo, side_coords, t2s_map, s2t_map);

    // print out results
    std::cout << mesh.to_yaml() << std::endl;
    END_EXAMPLE("blueprint_demo_transform_side");

    // print out t2s_map and s2t_map
    std::cout << t2s_map.to_yaml() << std::endl;
    std::cout << s2t_map.to_yaml() << std::endl;

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "uniform",
          "dims": 
          {
            "i": 3,
            "j": 3
          },
          "origin": 
          {
            "x": -10.0,
            "y": -10.0
          },
          "spacing": 
          {
            "dx": 10.0,
            "dy": 10.0
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "uniform",
          "coordset": "coords"
        }
      },
      "fields": 
      {
        "field": 
        {
          "association": "element",
          "topology": "mesh",
          "volume_dependent": "false",
          "values": [0.0, 1.0, 2.0, 3.0]
        }
      }
    }
    )";

    validate_basic_example("uniform",mesh,mesh_json);
}

