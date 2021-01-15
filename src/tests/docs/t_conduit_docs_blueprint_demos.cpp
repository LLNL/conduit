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
#include "conduit_relay.hpp"
#include "t_conduit_docs_tutorial_helpers.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

/// Helper Functions ///

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

void validate_basic_example(const std::string &name,
                            const Node &mesh,
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

    conduit::relay::io_blueprint::save(mesh, "basic_"+name+".blueprint_root");
    // conduit::relay::io_blueprint::save(mesh, "basic_"+name+".blueprint_root_hdf5");
}

/// Test Functions ///

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_uniform)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_uniform");
    // create container node
    Node mesh;
    // generate simple uniform 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("uniform", 3, 3, 0, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_uniform");

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

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_rectilinear)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_rectilinear");
    // create container node
    Node mesh;
    // generate simple rectilinear 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("rectilinear", 3, 3, 0, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_rectilinear");

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "rectilinear",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0],
            "y": [-10.0, 0.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "rectilinear",
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

    validate_basic_example("rectilinear",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_structured)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_structured");
    // create container node
    Node mesh;
    // generate simple structured 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("structured", 3, 3, 1, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_structured");

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "explicit",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
            "y": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "structured",
          "coordset": "coords",
          "elements": 
          {
            "dims": 
            {
              "i": 2,
              "j": 2
            }
          }
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

    validate_basic_example("structured",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_tris)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_tris");
    // create container node
    Node mesh;
    // generate simple explicit tri-based 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("tris", 3, 3, 0, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_tris");

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "explicit",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
            "y": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "unstructured",
          "coordset": "coords",
          "elements": 
          {
            "shape": "tri",
            "connectivity": [0, 3, 4, 0, 1, 4, 1, 4, 5, 1, 2, 5, 3, 6, 7, 3, 4, 7, 4, 7, 8, 4, 5, 8]
          }
        }
      },
      "fields": 
      {
        "field": 
        {
          "association": "element",
          "topology": "mesh",
          "volume_dependent": "false",
          "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        }
      }
    }
    )";

    validate_basic_example("tris",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_quads)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_quads");
    // create container node
    Node mesh;
    // generate simple explicit quad-based 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("quads", 3, 3, 0, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_quads");

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "explicit",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
            "y": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "unstructured",
          "coordset": "coords",
          "elements": 
          {
            "shape": "quad",
            "connectivity": [0, 3, 4, 1, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5]
          }
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

    validate_basic_example("quads",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_tets)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_tets");
    // create container node
    Node mesh;
    // generate simple explicit tri-based 3d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("tets", 3, 3, 3, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_tets");
    
    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "explicit",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
            "y": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "z": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "unstructured",
          "coordset": "coords",
          "elements": 
          {
            "shape": "tet",
            "connectivity": [0, 4, 1, 13, 0, 3, 4, 13, 0, 12, 3, 13, 0, 9, 12, 13, 0, 10, 9, 13, 0, 1, 10, 13, 1, 5, 2, 14, 1, 4, 5, 14, 1, 13, 4, 14, 1, 10, 13, 14, 1, 11, 10, 14, 1, 2, 11, 14, 3, 7, 4, 16, 3, 6, 7, 16, 3, 15, 6, 16, 3, 12, 15, 16, 3, 13, 12, 16, 3, 4, 13, 16, 4, 8, 5, 17, 4, 7, 8, 17, 4, 16, 7, 17, 4, 13, 16, 17, 4, 14, 13, 17, 4, 5, 14, 17, 9, 13, 10, 22, 9, 12, 13, 22, 9, 21, 12, 22, 9, 18, 21, 22, 9, 19, 18, 22, 9, 10, 19, 22, 10, 14, 11, 23, 10, 13, 14, 23, 10, 22, 13, 23, 10, 19, 22, 23, 10, 20, 19, 23, 10, 11, 20, 23, 12, 16, 13, 25, 12, 15, 16, 25, 12, 24, 15, 25, 12, 21, 24, 25, 12, 22, 21, 25, 12, 13, 22, 25, 13, 17, 14, 26, 13, 16, 17, 26, 13, 25, 16, 26, 13, 22, 25, 26, 13, 23, 22, 26, 13, 14, 23, 26]
          }
        }
      },
      "fields": 
      {
        "field": 
        {
          "association": "element",
          "topology": "mesh",
          "volume_dependent": "false",
          "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0]
        }
      }
    }
    )";

    validate_basic_example("tets",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_hexs)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_hexs");
    // create container node
    Node mesh;
    // generate simple explicit quad-based 3d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("hexs", 3, 3, 3, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_hexs");

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "explicit",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
            "y": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "z": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "unstructured",
          "coordset": "coords",
          "elements": 
          {
            "shape": "hex",
            "connectivity": [0, 1, 4, 3, 9, 10, 13, 12, 1, 2, 5, 4, 10, 11, 14, 13, 3, 4, 7, 6, 12, 13, 16, 15, 4, 5, 8, 7, 13, 14, 17, 16, 9, 10, 13, 12, 18, 19, 22, 21, 10, 11, 14, 13, 19, 20, 23, 22, 12, 13, 16, 15, 21, 22, 25, 24, 13, 14, 17, 16, 22, 23, 26, 25]
          }
        }
      },
      "fields": 
      {
        "field": 
        {
          "association": "element",
          "topology": "mesh",
          "volume_dependent": "false",
          "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        }
      }
    }
    )";

    validate_basic_example("hexs",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_polygons)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_polygons");
    // create container node
    Node mesh;
    // generate simple explicit poly-based 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("polygons", 3, 3, 0, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_polygons");

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "explicit",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
            "y": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "unstructured",
          "coordset": "coords",
          "elements": 
          {
            "shape": "polygonal",
            "connectivity": [0, 3, 4, 1, 1, 4, 5, 2, 3, 6, 7, 4, 4, 7, 8, 5],
            "sizes": [4, 4, 4, 4],
            "offsets": [0, 4, 8, 12]
          }
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

    validate_basic_example("polygons",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_polyhedra)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_polyhedra");
    // create container node
    Node mesh;
    // generate simple explicit poly-based 3d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("polyhedra", 3, 3, 3, mesh);
    // print out results
    mesh.print();
    END_EXAMPLE("blueprint_demo_basic_polyhedra");

    const std::string mesh_json = R"(
    {
      "coordsets": 
      {
        "coords": 
        {
          "type": "explicit",
          "values": 
          {
            "x": [-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0],
            "y": [-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "z": [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
          }
        }
      },
      "topologies": 
      {
        "mesh": 
        {
          "type": "unstructured",
          "coordset": "coords",
          "elements": 
          {
            "shape": "polyhedral",
            "connectivity": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 10, 11, 3, 12, 13, 14, 15, 16, 9, 17, 18, 12, 19, 5, 20, 21, 22, 23, 24, 10, 25, 26, 27, 21, 28, 15, 22, 29, 30, 31, 32, 19, 27, 33, 34, 29, 35],
            "sizes": [6, 6, 6, 6, 6, 6, 6, 6],
            "offsets": [0, 6, 12, 18, 24, 30, 36, 42]
          },
          "subelements": 
          {
            "shape": "polygonal",
            "connectivity": [0, 3, 4, 1, 0, 1, 10, 9, 1, 4, 13, 10, 4, 3, 12, 13, 3, 0, 9, 12, 9, 10, 13, 12, 1, 4, 5, 2, 1, 2, 11, 10, 2, 5, 14, 11, 5, 4, 13, 14, 10, 11, 14, 13, 3, 6, 7, 4, 4, 7, 16, 13, 7, 6, 15, 16, 6, 3, 12, 15, 12, 13, 16, 15, 4, 7, 8, 5, 5, 8, 17, 14, 8, 7, 16, 17, 13, 14, 17, 16, 9, 10, 19, 18, 10, 13, 22, 19, 13, 12, 21, 22, 12, 9, 18, 21, 18, 19, 22, 21, 10, 11, 20, 19, 11, 14, 23, 20, 14, 13, 22, 23, 19, 20, 23, 22, 13, 16, 25, 22, 16, 15, 24, 25, 15, 12, 21, 24, 21, 22, 25, 24, 14, 17, 26, 23, 17, 16, 25, 26, 22, 23, 26, 25],
            "sizes": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140]
          }
        }
      },
      "fields": 
      {
        "field": 
        {
          "association": "element",
          "topology": "mesh",
          "volume_dependent": "false",
          "values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        }
      }
    }
    )";

    validate_basic_example("polyhedra",mesh,mesh_json);
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_basic_uniform_complete)
{
    BEGIN_EXAMPLE("blueprint_demo_basic_uniform_complete");
    // create a Conduit node to hold our mesh data
    Node mesh;
    
    // create the coordinate set
    mesh["coordsets/coords/type"] = "uniform";
    mesh["coordsets/coords/dims/i"] = 3;
    mesh["coordsets/coords/dims/j"] = 3;
    // add origin and spacing to the coordset (optional)
    mesh["coordsets/coords/origin/x"] = -10.0;
    mesh["coordsets/coords/origin/y"] = -10.0;
    mesh["coordsets/coords/spacing/dx"] = 10.0;
    mesh["coordsets/coords/spacing/dy"] = 10.0;
    
    // add the topology
    // this case is simple b/c it's implicitly derived from the coordinate set
    mesh["topologies/topo/type"] = "uniform";
    // reference the coordinate set by name
    mesh["topologies/topo/coordset"] = "coords";
    
    // add a simple element-associated field 
    mesh["fields/ele_example/association"] =  "element";
    // reference the topology this field is defined on by name
    mesh["fields/ele_example/topology"] =  "topo";
    // set the field values, for this case we have 4 elements
    mesh["fields/ele_example/values"].set(DataType::float64(4));
    
    float64 *ele_vals_ptr = mesh["fields/ele_example/values"].value();
    
    for(int i=0;i<4;i++)
    {
        ele_vals_ptr[i] = float64(i);
    }
    
    // add a simple vertex-associated field 
    mesh["fields/vert_example/association"] =  "vertex";
    // reference the topology this field is defined on by name
    mesh["fields/vert_example/topology"] =  "topo";
    // set the field values, for this case we have 9 vertices
    mesh["fields/vert_example/values"].set(DataType::float64(9));
    
    float64 *vert_vals_ptr = mesh["fields/vert_example/values"].value();
    
    for(int i=0;i<9;i++)
    {
        vert_vals_ptr[i] = float64(i);
    }
    
    // make sure we conform:
    Node verify_info;
    if(!blueprint::mesh::verify(mesh, verify_info))
    {
        std::cout << "Verify failed!" << std::endl;
        verify_info.print();
    }

    // print out results
    mesh.print();
    
    // save our mesh to a file that can be read by VisIt
    //
    // this will create the file: complete_uniform_mesh_example.root
    // which includes the mesh blueprint index and the mesh data
    conduit::relay::io::blueprint::write_mesh(mesh,
                                              "complete_uniform_mesh_example",
                                              "json");

    END_EXAMPLE("blueprint_demo_basic_uniform_complete");
}
