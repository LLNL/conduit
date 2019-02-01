//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: t_blueprint_mesh_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

index_t OUTPUT_NUM_AXIS_POINTS = 5;

std::string PROTOCOL_VER = CONDUIT_VERSION;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_2d)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);

    bool silo_enabled = io_protos["io/protocols/conduit_silo"].as_string() == "enabled";
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";

    // we are using one node to hold group of example meshes purely out of convenience  
    Node dsets;
    // can be overridden via command line
    index_t npts_x = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_y = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_z = 1; // 2D examples ...

    blueprint::mesh::examples::braid("uniform",
                                      npts_x,
                                      npts_y,
                                      npts_z,
                                      dsets["uniform"]);

    blueprint::mesh::examples::braid("rectilinear",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["rect"]);

    blueprint::mesh::examples::braid("structured",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["struct"]);

    blueprint::mesh::examples::braid("lines",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["lines"]);

    blueprint::mesh::examples::braid("tris",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["tris"]);

    blueprint::mesh::examples::braid("quads",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["quads"]);

    blueprint::mesh::examples::braid("quads_and_tris",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["quads_and_tris"]);

    blueprint::mesh::examples::braid("quads_and_tris_offsets",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["quads_and_tris_offsets"]);

    blueprint::mesh::examples::braid("points",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points"]);

    blueprint::mesh::examples::braid("points_implicit",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points_implicit"]);

    Node info;
    NodeConstIterator itr = dsets.children();
    while(itr.has_next())
    {
        const Node &mesh = itr.next();
        EXPECT_TRUE(blueprint::mesh::verify(mesh,info));
        CONDUIT_INFO(info.to_json());
    }

    // TODO: Add VisIt support for rendering mixed element and implicit point
    // meshes so they don't have to be removed before outputting mesh data.
    dsets.remove("quads_and_tris");
    dsets.remove("quads_and_tris_offsets");
    dsets.remove("points_implicit");

    relay::io_blueprint::save(dsets, "braid_2d_examples.blueprint_root");
    if(hdf5_enabled)
    {
        relay::io_blueprint::save(dsets, "braid_2d_examples.blueprint_root_hdf5");
    }
    if(silo_enabled)
    {
        // we removed datasets above, so we need an updated iterator
        itr = dsets.children();
        while(itr.has_next())
        {
            const Node &mesh = itr.next();
            std::string name = itr.name();

            // Skip output of silo mesh for mixed mesh of tris and quads for now.
            // The silo output is not yet defined and it throws an exception
            // in conduit_silo.cpp::silo_write_ucd_zonelist()
            // in the following line that is looking for the 'shape' node:
            //              std::string topo_shape = shape_block->fetch("shape").as_string();
            // which does not exist for indexed_stream meshes.
            // The silo writer needs to be updated for this case.
            if( name == "quads_and_tris" || name == "quads_and_tris_offsets" )
            {
                CONDUIT_INFO("\tNOTE: skipping output to SILO -- ")
                CONDUIT_INFO("feature is unavailable for mixed element meshes")
                continue;
            }

            relay::io::save(mesh,
                            "braid_2d_" + name +  "_example.silo:mesh",
                            "conduit_silo_mesh");
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_3d)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);

    bool silo_enabled = io_protos["io/protocols/conduit_silo"].as_string() == "enabled";
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";

    // we are using one node to hold group of example meshes purely out of convenience  
    Node dsets;
    // can be overridden via command line
    index_t npts_x = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_y = OUTPUT_NUM_AXIS_POINTS;
    index_t npts_z = OUTPUT_NUM_AXIS_POINTS; // 3D examples ...

    blueprint::mesh::examples::braid("uniform",
                                      npts_x,
                                      npts_y,
                                      npts_z,
                                      dsets["uniform"]);

    blueprint::mesh::examples::braid("rectilinear",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["rect"]);

    blueprint::mesh::examples::braid("structured",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["struct"]);

    blueprint::mesh::examples::braid("points",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points"]);

    blueprint::mesh::examples::braid("points_implicit",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points_implicit"]);

    blueprint::mesh::examples::braid("lines",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["lines"]);

    blueprint::mesh::examples::braid("tets",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["tets"]);

    blueprint::mesh::examples::braid("hexs",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["hexs"]);

    blueprint::mesh::examples::braid("hexs_and_tets",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["hexs_and_tets"]);

    Node info;
    NodeConstIterator itr = dsets.children();
    while(itr.has_next())
    {
        const Node &mesh = itr.next();
        EXPECT_TRUE(blueprint::mesh::verify(mesh,info));
        CONDUIT_INFO(info.to_json());
    }

    // TODO: Add VisIt support for rendering mixed element and implicit point
    // meshes so they don't have to be removed before outputting mesh data.
    dsets.remove("hexs_and_tets");
    dsets.remove("points_implicit");

    relay::io_blueprint::save(dsets, "braid_3d_examples.blueprint_root");
    if(hdf5_enabled)
    {
        relay::io_blueprint::save(dsets, "braid_3d_examples.blueprint_root_hdf5");
    }
    if(silo_enabled)
    {
        // we removed datasets above, so we need an updated iterator
        itr = dsets.children();
        while(itr.has_next())
        {
            const Node &mesh = itr.next();
            std::string name = itr.name();

            // Skip output of silo mesh for mixed mesh of hexs and tets for now.
            // The silo output is not yet defined and it throws an exception
            // in conduit_silo.cpp::silo_write_ucd_zonelist()
            // in the following line that is looking for the 'shape' node:
            //              std::string topo_shape = shape_block->fetch("shape").as_string();
            // which does not exist for indexed_stream meshes.
            // The silo writer needs to be updated for this case.
            if(name == "hexs_and_tets")
            {
                CONDUIT_INFO("\tNOTE: skipping output to SILO -- ")
                CONDUIT_INFO("feature is unavailable for mixed element meshes")
                continue;
            }

            relay::io::save(mesh,
                            "braid_3d_" + name +  "_example.silo:mesh",
                            "conduit_silo_mesh");
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, julia)
{
    Node res;
    blueprint::mesh::examples::julia(500,   500, // nx, ny
                                     -2.0,  2.0, // x range
                                     -2.0,  2.0, // y range
                                     0.285, 0.01, // c value
                                     res);
    Node info;
    EXPECT_TRUE(blueprint::mesh::verify(res,info));
    CONDUIT_INFO(info.to_json());

    relay::io_blueprint::save(res, "julia_example.blueprint_root");
}



//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, spiral)
{
    int ndoms = 10;
    Node res;
    blueprint::mesh::examples::spiral(ndoms,res["spiral"]);
    Node info;
    EXPECT_TRUE(blueprint::mesh::verify(res["spiral/domain_000000"],info));
    CONDUIT_INFO(info.to_json());

    blueprint::mesh::generate_index(res["spiral/domain_000000"],
                                    "",
                                    ndoms,
                                    res["blueprint_index/spiral"]);

    // save json
    res["protocol/name"] = "json";
    res["protocol/version"] = PROTOCOL_VER;

    res["number_of_files"] = 1;
    res["number_of_trees"] = ndoms;
    res["file_pattern"] = "spiral_example.blueprint_root";
    res["tree_pattern"] = "spiral/domain_%06d";

    CONDUIT_INFO("Creating: spiral_example.blueprint_root")
    relay::io::save(res,"spiral_example.blueprint_root","json");
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, polytess)
{
    const index_t nlevels = 3;
    Node res;
    blueprint::mesh::examples::polytess(nlevels,
                                        res);

    Node info;
    EXPECT_TRUE(blueprint::mesh::verify(res,info));
    CONDUIT_INFO(info.to_json());

    relay::io_blueprint::save(res, "polytess_example.blueprint_root");
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, 2d_braid_zero_z_check)
{
    Node mesh, info;
    // these checks make sure braid generates valid fields even when
    // # of z pointers == 0
    int npts_x = 5;
    int npts_y = 5;
    int npts_z = 0;

    std::vector<std::string> braid_type_strings;
    braid_type_strings.push_back("points");
    braid_type_strings.push_back("points_implicit");
    braid_type_strings.push_back("lines");
    braid_type_strings.push_back("rectilinear");
    braid_type_strings.push_back("structured");
    braid_type_strings.push_back("tris");
    braid_type_strings.push_back("quads");
    
    for(index_t i = 0; i < braid_type_strings.size(); i++)
    {
        mesh.reset();
        blueprint::mesh::examples::braid(braid_type_strings[i],
                                          npts_x,
                                          npts_y,
                                          npts_z,
                                          mesh);
        // make the braid vertex-assoced field has with more than zero entries
        EXPECT_GT(mesh["fields/braid/values"].dtype().number_of_elements(),0);
        mesh.print();
    }
}


//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    
    // allow override of the data size via the command line
    if(argc == 2)
    { 
        OUTPUT_NUM_AXIS_POINTS = atoi(argv[1]);
    }
    
    result = RUN_ALL_TESTS();
    return result;
}
