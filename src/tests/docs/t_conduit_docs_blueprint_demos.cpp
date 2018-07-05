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

void convert_schema_to_mesh(const std::string &schema, conduit::Node &mesh)
{
    mesh.reset();
    conduit::Generator gen(schema, "json", NULL);
    gen.walk(mesh);
}

void expand_mesh_data(const conduit::Node &mesh, conduit::Node &emesh)
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
            expand_mesh_data(child, echild);
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

void save_mesh(const conduit::Node &mesh, const std::string &mesh_name)
{
    Node mesh_ctrl, mesh_root;
    Node &mesh_index = mesh_root["blueprint_index"];
    {
        mesh_ctrl[mesh_name].set_external(mesh);
        blueprint::mesh::generate_index(mesh, mesh_name, 1, mesh_index[mesh_name]);

        mesh_root["protocol/name"] = "conduit_hdf5";
        mesh_root["protocol/version"] = CONDUIT_VERSION;
        mesh_root["number_of_files"] = 1;
        mesh_root["number_of_trees"] = 1;
        mesh_root["file_pattern"] = mesh_name + ".hdf5";
        mesh_root["tree_pattern"] = "";

        relay::io::save(mesh_root,mesh_name + ".blueprint_root_hdf5","hdf5");
        relay::io::save(mesh_ctrl,mesh_name + ".hdf5","hdf5");
    }
}

/// Test Functions ///

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_gradient_uniform)
{
    // start demo code

    // create container node and debug node
    Node mesh, info;

    // generate simple uniform 2d 'gradient' mesh
    conduit::blueprint::mesh::examples::gradient("uniform", 3, 3, 0, mesh);

    // start demo string

    const std::string mesh_schema = R"(
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

    // start test code

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node wide_mesh, ref_mesh;
    convert_schema_to_mesh(mesh_schema, ref_mesh);
    expand_mesh_data(mesh, wide_mesh);
    EXPECT_FALSE(wide_mesh.diff(ref_mesh, info));

    // TODO(JRC): Output and validate the saved files.
    // save_mesh(mesh, "gradient_uniform");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_gradient_rectilinear)
{
    // start demo code

    // create container node and debug node
    Node mesh, info;

    // generate simple rectilinear 2d 'gradient' mesh
    conduit::blueprint::mesh::examples::gradient("rectilinear", 3, 3, 0, mesh);

    // start demo string

    const std::string mesh_schema = R"(
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

    // start test code

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node wide_mesh, ref_mesh;
    convert_schema_to_mesh(mesh_schema, ref_mesh);
    expand_mesh_data(mesh, wide_mesh);
    EXPECT_FALSE(wide_mesh.diff(ref_mesh, info));

    // TODO(JRC): Expect that the proper files exist.
    // save_mesh(mesh, "gradient_rectilinear");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_gradient_structured)
{
    // start demo code

    // create container node and debug node
    Node mesh, info;

    // generate simple structured 2d 'gradient' mesh
    conduit::blueprint::mesh::examples::gradient("structured", 3, 3, 1, mesh);

    // start demo string

    const std::string mesh_schema = R"(
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

    // start test code

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node wide_mesh, ref_mesh;
    convert_schema_to_mesh(mesh_schema, ref_mesh);
    expand_mesh_data(mesh, wide_mesh);
    EXPECT_FALSE(wide_mesh.diff(ref_mesh, info));

    // TODO(JRC): Expect that the proper files exist.
    // save_mesh(mesh, "gradient_structured");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_gradient_tris)
{
    // start demo code

    // create container node and debug node
    Node mesh, info;

    // generate simple uniform 2d 'gradient' mesh
    conduit::blueprint::mesh::examples::gradient("tris", 3, 3, 0, mesh);

    // start demo string

    const std::string mesh_schema = R"(
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

    // start test code

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node wide_mesh, ref_mesh;
    convert_schema_to_mesh(mesh_schema, ref_mesh);
    expand_mesh_data(mesh, wide_mesh);
    EXPECT_FALSE(wide_mesh.diff(ref_mesh, info));

    // TODO(JRC): Expect that the proper files exist.
    // save_mesh(mesh, "gradient_tris");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_gradient_quads)
{
    // start demo code

    // create container node and debug node
    Node mesh, info;

    // generate simple uniform 2d 'gradient' mesh
    conduit::blueprint::mesh::examples::gradient("quads", 3, 3, 0, mesh);

    // start demo string

    const std::string mesh_schema = R"(
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

    // start test code

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node wide_mesh, ref_mesh;
    convert_schema_to_mesh(mesh_schema, ref_mesh);
    expand_mesh_data(mesh, wide_mesh);
    EXPECT_FALSE(wide_mesh.diff(ref_mesh, info));

    // TODO(JRC): Expect that the proper files exist.
    // save_mesh(mesh, "gradient_quads");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_gradient_tets)
{
    // start demo code

    // create container node and debug node
    Node mesh, info;

    // generate simple uniform 2d 'gradient' mesh
    conduit::blueprint::mesh::examples::gradient("tets", 3, 3, 3, mesh);

    // start demo string

    const std::string mesh_schema = R"(
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

    // start test code

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node wide_mesh, ref_mesh;
    convert_schema_to_mesh(mesh_schema, ref_mesh);
    expand_mesh_data(mesh, wide_mesh);
    EXPECT_FALSE(wide_mesh.diff(ref_mesh, info));

    // TODO(JRC): Expect that the proper files exist.
    // save_mesh(mesh, "gradient_tets");
}

//-----------------------------------------------------------------------------
TEST(conduit_docs, blueprint_demo_gradient_hexs)
{
    // start demo code

    // create container node and debug node
    Node mesh, info;

    // generate simple uniform 2d 'gradient' mesh
    conduit::blueprint::mesh::examples::gradient("hexs", 3, 3, 3, mesh);

    // start demo string

    const std::string mesh_schema = R"(
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

    // start test code

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node wide_mesh, ref_mesh;
    convert_schema_to_mesh(mesh_schema, ref_mesh);
    expand_mesh_data(mesh, wide_mesh);
    EXPECT_FALSE(wide_mesh.diff(ref_mesh, info));

    // TODO(JRC): Expect that the proper files exist.
    // save_mesh(mesh, "gradient_hexs");
}
