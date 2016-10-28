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
/// file: conduit_blueprint_mesh_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

index_t OUTPUT_NUM_AXIS_POINTS = 5;


//-----------------------------------------------------------------------------
void
create_blueprint_index_for_3d_examples(Node &index_root)
{
    // uniform
        Node &uni_idx = index_root["uniform"];
        // state
        uni_idx["state/cycle"] = 42;
        uni_idx["state/time"]  = 3.1415;
        uni_idx["state/number_of_domains"]  = 1;
        // coords
        uni_idx["coordsets/coords/type"]         = "uniform";
        uni_idx["coordsets/coords/coord_system"] = "xyz";
        uni_idx["coordsets/coords/path"]         = "uniform/coordsets/coords";
        // topology
        uni_idx["topologies/mesh/type"]     = "uniform";
        uni_idx["topologies/mesh/coordset"] = "coords";
        uni_idx["topologies/mesh/path"]     = "uniform/topologies/mesh";
        // fields
            // pc
            uni_idx["fields/braid/number_of_components"] = 1;
            uni_idx["fields/braid/association"] = "point";
            uni_idx["fields/braid/topology"]    = "mesh";
            uni_idx["fields/braid/path"]        = "uniform/fields/braid";
            // ec
            uni_idx["fields/radial/number_of_components"] = 1;
            uni_idx["fields/radial/association"] = "element";
            uni_idx["fields/radial/topology"]    = "mesh";
            uni_idx["fields/radial/path"]        = "uniform/fields/radial";
            // vel pc
            uni_idx["fields/vel/number_of_components"] = 3;
            uni_idx["fields/vel/association"] = "point";
            uni_idx["fields/vel/topology"]    = "mesh";
            uni_idx["fields/vel/path"]        = "uniform/fields/vel";


    // rectilinear
        Node &rect_idx = index_root["rect"];
        // state
        rect_idx["state/cycle"] = 42;
        rect_idx["state/time"]  = 3.1415;
        rect_idx["state/number_of_domains"]  = 1;
        // coords
        rect_idx["coordsets/coords/type"]         = "rectilinear";
        rect_idx["coordsets/coords/coord_system"] = "xyz";
        rect_idx["coordsets/coords/path"]         = "rect/coordsets/coords";
        // topology
        rect_idx["topologies/mesh/type"]     = "rectilinear";
        rect_idx["topologies/mesh/coordset"] = "coords";
        rect_idx["topologies/mesh/path"]     = "rect/topologies/mesh";
        // fields
            // pc
            rect_idx["fields/braid/number_of_components"] = 1;
            rect_idx["fields/braid/association"] = "point";
            rect_idx["fields/braid/topology"]    = "mesh";
            rect_idx["fields/braid/path"]        = "rect/fields/braid";
            // ec
            rect_idx["fields/radial/number_of_components"] = 1;
            rect_idx["fields/radial/association"] = "element";
            rect_idx["fields/radial/topology"]    = "mesh";
            rect_idx["fields/radial/path"]        = "rect/fields/radial";
            // vel pc
            rect_idx["fields/vel/number_of_components"] = 3;
            rect_idx["fields/vel/association"] = "point";
            rect_idx["fields/vel/topology"]    = "mesh";
            rect_idx["fields/vel/path"]        = "rect/fields/vel";


    // structured
        Node &struct_idx = index_root["struct"];
        // state
        struct_idx["state/cycle"] = 42;
        struct_idx["state/time"]  = 3.1415;
        struct_idx["state/number_of_domains"]  = 1;
        // coords
        struct_idx["coordsets/coords/type"]         = "explicit";
        struct_idx["coordsets/coords/coord_system"] = "xyz";
        struct_idx["coordsets/coords/path"]         = "struct/coordsets/coords";
        // topology
        struct_idx["topologies/mesh/type"]     = "structured";
        struct_idx["topologies/mesh/coordset"] = "coords";
        struct_idx["topologies/mesh/path"]     = "struct/topologies/mesh";
        // fields
            // pc
            struct_idx["fields/braid/number_of_components"] = 1;
            struct_idx["fields/braid/association"] = "point";
            struct_idx["fields/braid/topology"]    = "mesh";
            struct_idx["fields/braid/path"]        = "struct/fields/braid";
            // ec
            struct_idx["fields/radial/number_of_components"] = 1;
            struct_idx["fields/radial/association"] = "element";
            struct_idx["fields/radial/topology"]    = "mesh";
            struct_idx["fields/radial/path"]        = "struct/fields/radial";
            // vel pc
            struct_idx["fields/vel/number_of_components"] = 3;
            struct_idx["fields/vel/association"] = "point";
            struct_idx["fields/vel/topology"]    = "mesh";
            struct_idx["fields/vel/path"]        = "struct/fields/vel";


    // lines (unstructured)
        Node &lines_idx = index_root["lines"];
        // state
        lines_idx["state/cycle"] = 42;
        lines_idx["state/time"]  = 3.1415;
        lines_idx["state/number_of_domains"]  = 1;
        // coords
        lines_idx["coordsets/coords/type"]         = "explicit";
        lines_idx["coordsets/coords/coord_system"] = "xyz";
        lines_idx["coordsets/coords/path"]         = "lines/coordsets/coords";
        // topology
        lines_idx["topologies/mesh/type"]     = "unstructured";
        lines_idx["topologies/mesh/coordset"] = "coords";
        lines_idx["topologies/mesh/path"]     = "lines/topologies/mesh";
        // fields
            // pc
            lines_idx["fields/braid/number_of_components"] = 1;
            lines_idx["fields/braid/association"] = "point";
            lines_idx["fields/braid/topology"]    = "mesh";
            lines_idx["fields/braid/path"]   = "lines/fields/braid";
            // ec
            lines_idx["fields/radial/number_of_components"] = 1;
            lines_idx["fields/radial/association"] = "element";
            lines_idx["fields/radial/topology"]    = "mesh";
            lines_idx["fields/radial/path"]        = "lines/fields/radial";
            // vel pc
            lines_idx["fields/vel/number_of_components"] = 3;
            lines_idx["fields/vel/association"] = "point";
            lines_idx["fields/vel/topology"]    = "mesh";
            lines_idx["fields/vel/path"]        = "lines/fields/vel";

    // tets (unstructured)
    Node &tets_idx = index_root["tets"];
    // state
    tets_idx["state/cycle"] = 42;
    tets_idx["state/time"]  = 3.1415;
    tets_idx["state/number_of_domains"]  = 1;
    // coords
    tets_idx["coordsets/coords/type"]         = "explicit";
    tets_idx["coordsets/coords/coord_system"] = "xyz";
    tets_idx["coordsets/coords/path"]         = "tets/coordsets/coords";
    // topology
    tets_idx["topologies/mesh/type"]     = "unstructured";
    tets_idx["topologies/mesh/coordset"] = "coords";
    tets_idx["topologies/mesh/path"]     = "tets/topologies/mesh";
    // fields
        // pc
        tets_idx["fields/braid/number_of_components"] = 1;
        tets_idx["fields/braid/association"] = "point";
        tets_idx["fields/braid/topology"]    = "mesh";
        tets_idx["fields/braid/path"]        = "tets/fields/braid";
        // ec
        tets_idx["fields/radial/number_of_components"] = 1;
        tets_idx["fields/radial/association"] = "element";
        tets_idx["fields/radial/topology"]    = "mesh";
        tets_idx["fields/radial/path"]        = "tets/fields/radial";
        // vel pc
        tets_idx["fields/vel/number_of_components"] = 3;
        tets_idx["fields/vel/association"] = "point";
        tets_idx["fields/vel/topology"]    = "mesh";
        tets_idx["fields/vel/path"]        = "tets/fields/vel";

    
    // hexs (unstructured)
    Node &hexs_idx = index_root["hexs"];
    // state
    hexs_idx["state/cycle"] = 42;
    hexs_idx["state/time"]  = 3.1415;
    hexs_idx["state/number_of_domains"]  = 1;
    // coords
    hexs_idx["coordsets/coords/type"]         = "explicit";
    hexs_idx["coordsets/coords/coord_system"] = "xyz";
    hexs_idx["coordsets/coords/path"]         = "hexs/coordsets/coords";
    // topology
    hexs_idx["topologies/mesh/type"]     = "unstructured";
    hexs_idx["topologies/mesh/coordset"] = "coords";
    hexs_idx["topologies/mesh/path"]     = "hexs/topologies/mesh";
    // fields
        // pc
        hexs_idx["fields/braid/number_of_components"] = 1;
        hexs_idx["fields/braid/association"] = "point";
        hexs_idx["fields/braid/topology"]    = "mesh";
        hexs_idx["fields/braid/path"]        = "hexs/fields/braid";
        // ec
        hexs_idx["fields/radial/number_of_components"] = 1;
        hexs_idx["fields/radial/association"] = "element";
        hexs_idx["fields/radial/topology"]    = "mesh";
        hexs_idx["fields/radial/path"]        = "hexs/fields/radial";
        // vel pc
        hexs_idx["fields/vel/number_of_components"] = 3;
        hexs_idx["fields/vel/association"] = "point";
        hexs_idx["fields/vel/topology"]    = "mesh";
        hexs_idx["fields/vel/path"]        = "hexs/fields/vel";


    // points (unstructured)
    Node &pts_idx = index_root["points"];
    // state
    pts_idx["state/cycle"] = 42;
    pts_idx["state/time"]  = 3.1415;
    pts_idx["state/number_of_domains"]  = 1;
    // coords
    pts_idx["coordsets/coords/type"]         = "explicit";
    pts_idx["coordsets/coords/coord_system"] = "xyz";
    pts_idx["coordsets/coords/path"]         = "points/coordsets/coords";
    // topology
    pts_idx["topologies/mesh/type"]     = "unstructured";
    pts_idx["topologies/mesh/coordset"] = "coords";
    pts_idx["topologies/mesh/path"]     = "points/topologies/mesh";
    // fields
        // pc
        pts_idx["fields/braid/number_of_components"] = 1;
        pts_idx["fields/braid/association"] = "point";
        pts_idx["fields/braid/topology"]    = "mesh";
        pts_idx["fields/braid/path"]        = "points/fields/braid";
        // ec
        pts_idx["fields/radial/number_of_components"] = 1;
        pts_idx["fields/radial/association"] = "element";
        pts_idx["fields/radial/topology"]    = "mesh";
        pts_idx["fields/radial/path"]        = "points/fields/radial";
        // vel pc
        pts_idx["fields/vel/number_of_components"] = 3;
        pts_idx["fields/vel/association"] = "point";
        pts_idx["fields/vel/topology"]    = "mesh";
        pts_idx["fields/vel/path"]        = "points/fields/vel";

}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_2d)
{
    Node io_protos;
    relay::about(io_protos);

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
                                      npts_y,
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
    
    NodeConstIterator itr = dsets.children();

    Node root;
    Node &bp_idx = root["blueprint_index"];
    while(itr.has_next())
    {
        Node info;
        const Node &mesh = itr.next();
        std::string mesh_name = itr.name();
        
        EXPECT_TRUE(blueprint::mesh::verify(mesh,info));
        CONDUIT_INFO(info.to_json());
        
        // skip unsupported types
        if( mesh_name != "quads_and_tris" || mesh_name != "quads_and_tris_offsets" )
        {
            CONDUIT_INFO("GEN MESH BP INDEX");        
            blueprint::mesh::generate_index(mesh,mesh_name,1,bp_idx[mesh_name]);
            CONDUIT_INFO(bp_idx[mesh_name].to_json());

            info.reset();
            EXPECT_TRUE(blueprint::mesh::index::verify(bp_idx[mesh_name],info));
            CONDUIT_INFO(info.to_json());
        }
    }
    
    
    if(silo_enabled)
    {
    
        NodeConstIterator itr = dsets.children();
    
        while(itr.has_next())
        {
            const Node &mesh = itr.next();
            std::string name = itr.name();
            CONDUIT_INFO("saving 2d example '" << name << "' to silo");
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
    
    if(hdf5_enabled)
    {
        
        //Node root;
        //create_blueprint_index_for_2d_examples(root["blueprint_index"]);

        Node info;
        NodeConstIterator idx_itr = root["blueprint_index"].children();
        while(idx_itr.has_next())
        {

            const Node &chld = idx_itr.next();
            EXPECT_TRUE(blueprint::mesh::index::verify(chld,info[idx_itr.name()]));
        }

        CONDUIT_INFO("blueprint::mesh::index verify info:");
        CONDUIT_INFO(info.to_json());
        
        root["protocol/name"] = "conduit_hdf5";
        root["protocol/version"] = "0.1";
        
        root["number_of_files"] = 1;
        root["number_of_trees"] = 1;
        root["file_pattern"] = "braid_2d_examples.hdf5";
        root["tree_pattern"] = "/";
        
        CONDUIT_INFO("Creating ")
        CONDUIT_INFO("Creating: braid_2d_examples.hdf5.blueprint_root")
        relay::io::save(root,"braid_2d_examples.blueprint_root_hdf5","hdf5");
        CONDUIT_INFO("Creating: braid_2d_examples.hdf5")
        relay::io::save(dsets,"braid_2d_examples.hdf5");
    }
    
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_3d)
{
    Node io_protos;
    relay::about(io_protos);

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
                                      npts_y,
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

    NodeConstIterator itr = dsets.children();

    Node root;
    Node &bp_idx = root["blueprint_index"];
    while(itr.has_next())
    {
        Node info;
        const Node &mesh = itr.next();
        std::string mesh_name = itr.name();
        
        EXPECT_TRUE(blueprint::mesh::verify(mesh,info));
        CONDUIT_INFO(info.to_json());
        
        // skip unsupported types
        if( mesh_name != "hexs_and_tets")
        {
            CONDUIT_INFO("GEN MESH BP INDEX");
            blueprint::mesh::generate_index(mesh,mesh_name,1,bp_idx[mesh_name]);
            CONDUIT_INFO(bp_idx[mesh_name].to_json());

            info.reset();
            EXPECT_TRUE(blueprint::mesh::index::verify(bp_idx[mesh_name],info));
            CONDUIT_INFO(info.to_json());
        }
        
    }

    if(silo_enabled)
    {
    
        itr.to_front();
        
        while(itr.has_next())
        {
            const Node &mesh = itr.next();
            //mesh.print();
            std::string name = itr.name();
            CONDUIT_INFO("saving 3d example '" << name << "' to silo")

            // Skip output of silo mesh for mixed mesh of hexs and tets for now.
            // The silo output is not yet defined and it throws an exception
            // in conduit_silo.cpp::silo_write_ucd_zonelist()
            // in the following line that is looking for the 'shape' node:
            //              std::string topo_shape = shape_block->fetch("shape").as_string();
            // which does not exist for indexed_stream meshes.
            // The silo writer needs to be updated for this case.
            if(name == "hexs_and_tets")
            {
                CONDUIT_INFO("\tskipping output to SILO -- this is not implemented yet for indexed_stream meshes.");
                continue;
            }
            relay::io::save(mesh,
                            "braid_3d_" + name +  "_example.silo:mesh",
                            "conduit_silo_mesh");
        }
    }
    
    if(hdf5_enabled)
    {
        //Node root;
        //create_blueprint_index_for_3d_examples(root["blueprint_index"]);
        
        Node info;
        NodeConstIterator idx_itr = root["blueprint_index"].children();
        while(idx_itr.has_next())
        {

            const Node &chld = idx_itr.next();
            EXPECT_TRUE(blueprint::mesh::index::verify(chld,info[idx_itr.name()]));
        }
        
        CONDUIT_INFO("blueprint::mesh::index verify info:");
        CONDUIT_INFO(info.to_json());

        root["protocol/name"]    = "hdf5";
        root["protocol/version"] = "0.1";
        
        root["number_of_files"] = 1;
        root["number_of_trees"] = 1;
        root["file_pattern"] = "braid_3d_examples.hdf5";
        root["tree_pattern"] = "/";

        CONDUIT_INFO("Creating: braid_3d_examples.hdf5.blueprint_root")
        relay::io::save(root,"braid_3d_examples.blueprint_root_hdf5","hdf5");
        CONDUIT_INFO("Creating: braid_3d_examples.hdf5")
        relay::io::save(dsets,"braid_3d_examples.hdf5");
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

