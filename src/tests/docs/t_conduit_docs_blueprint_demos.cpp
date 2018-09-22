//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_blueprint_demos.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

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
    // create container node
    Node mesh;
    // generate simple uniform 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("uniform", 3, 3, 0, mesh);
    // print out results
    mesh.print();

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
    // create container node
    Node mesh;
    // generate simple rectilinear 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("rectilinear", 3, 3, 0, mesh);
    // print out results
    mesh.print();

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
    // create container node
    Node mesh;
    // generate simple structured 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("structured", 3, 3, 1, mesh);
    // print out results
    mesh.print();

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
    // create container node
    Node mesh;
    // generate simple explicit tri-based 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("tris", 3, 3, 0, mesh);
    // print out results
    mesh.print();

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
    // create container node
    Node mesh;
    // generate simple explicit quad-based 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("quads", 3, 3, 0, mesh);
    // print out results
    mesh.print();

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
    // create container node
    Node mesh;
    // generate simple explicit tri-based 3d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("tets", 3, 3, 3, mesh);
    // print out results
    mesh.print();

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
    // create container node
    Node mesh;
    // generate simple explicit quad-based 3d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("hexs", 3, 3, 3, mesh);
    // print out results
    mesh.print();

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
    // create container node
    Node mesh;
    // generate simple explicit poly-based 2d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("polygons", 3, 3, 0, mesh);
    // print out results
    mesh.print();

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
            "connectivity": [4, 0, 3, 4, 1, 4, 1, 4, 5, 2, 4, 3, 6, 7, 4, 4, 4, 7, 8, 5]
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
    // create container node
    Node mesh;
    // generate simple explicit poly-based 3d 'basic' mesh
    conduit::blueprint::mesh::examples::basic("polyhedra", 3, 3, 3, mesh);
    // print out results
    mesh.print();

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
            "connectivity": [6, 4, 0, 4, 1, 3, 4, 0, 1, 9, 10, 4, 1, 3, 10, 12, 4, 0, 9, 4, 13, 4, 4, 13, 3, 12, 4, 9, 10, 13, 12, 6, 4, 1, 5, 2, 4, 4, 1, 2, 10, 11, 4, 2, 4, 11, 13, 4, 1, 10, 5, 14, 4, 5, 14, 4, 13, 4, 10, 11, 14, 13, 6, 4, 3, 7, 4, 6, 4, 3, 4, 12, 13, 4, 4, 6, 13, 15, 4, 3, 12, 7, 16, 4, 7, 16, 6, 15, 4, 12, 13, 16, 15, 6, 4, 4, 8, 5, 7, 4, 4, 5, 13, 14, 4, 5, 7, 14, 16, 4, 4, 13, 8, 17, 4, 8, 17, 7, 16, 4, 13, 14, 17, 16, 6, 4, 9, 13, 10, 12, 4, 9, 10, 18, 19, 4, 10, 12, 19, 21, 4, 9, 18, 13, 22, 4, 13, 22, 12, 21, 4, 18, 19, 22, 21, 6, 4, 10, 14, 11, 13, 4, 10, 11, 19, 20, 4, 11, 13, 20, 22, 4, 10, 19, 14, 23, 4, 14, 23, 13, 22, 4, 19, 20, 23, 22, 6, 4, 12, 16, 13, 15, 4, 12, 13, 21, 22, 4, 13, 15, 22, 24, 4, 12, 21, 16, 25, 4, 16, 25, 15, 24, 4, 21, 22, 25, 24, 6, 4, 13, 17, 14, 16, 4, 13, 14, 22, 23, 4, 14, 16, 23, 25, 4, 13, 22, 17, 26, 4, 17, 26, 16, 25, 4, 22, 23, 26, 25]
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
TEST(conduit_docs, blueprint_demo_basic_uniform_detailed)
{
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
    
    // save our mesh to a json that can be read by VisIt
    conduit::relay::io_blueprint::save(mesh, "basic_detailed_uniform.blueprint_root");
}
