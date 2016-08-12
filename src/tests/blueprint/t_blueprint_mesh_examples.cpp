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
#include "relay.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;


//-----------------------------------------------------------------------------
void
create_blueprint_index_for_2d_examples(Node &index_root)
{
    // uniform
        Node &uni_idx = index_root["uniform"];
        // state
        uni_idx["state/cycle"] = 42;
        uni_idx["state/time"]  = 3.1415;
        uni_idx["state/number_of_domains"]  = 1;
        // coords
        uni_idx["coordsets/coords/type"]         = "uniform";
        uni_idx["coordsets/coords/coord_system"] = "xy";
        uni_idx["coordsets/coords/path"]         = "uniform/coords";
        // topology
        uni_idx["topologies/mesh/type"]     = "uniform";
        uni_idx["topologies/mesh/coordset"] = "coords";
        uni_idx["topologies/mesh/path"]     = "uniform/topology";
        // fields
            // pc
            uni_idx["fields/braid_pc/number_of_components"] = 1;
            uni_idx["fields/braid_pc/association"] = "point";
            uni_idx["fields/braid_pc/topology"]    = "mesh";
            uni_idx["fields/braid_pc/path"]        = "uniform/fields/braid_pc";
            // ec
            uni_idx["fields/radial_ec/number_of_components"] = 1;
            uni_idx["fields/radial_ec/association"] = "element";
            uni_idx["fields/radial_ec/topology"]    = "mesh";
            uni_idx["fields/radial_ec/path"]        = "uniform/fields/radial_ec";

    // rectilinear
        Node &rect_idx = index_root["rect"];
        // state
        rect_idx["state/cycle"] = 42;
        rect_idx["state/time"]  = 3.1415;
        rect_idx["state/number_of_domains"]  = 1;
        // coords
        rect_idx["coordsets/coords/type"]         = "rectilinear";
        rect_idx["coordsets/coords/coord_system"] = "xy";
        rect_idx["coordsets/coords/path"]         = "rect/coords";
        // topology
        rect_idx["topologies/mesh/type"]     = "rectilinear";
        rect_idx["topologies/mesh/coordset"] = "coords";
        rect_idx["topologies/mesh/path"]     = "rect/topology";
        // fields
            // pc
            rect_idx["fields/braid_pc/number_of_components"] = 1;
            rect_idx["fields/braid_pc/association"] = "point";
            rect_idx["fields/braid_pc/topology"]    = "mesh";
            rect_idx["fields/braid_pc/path"]        = "rect/fields/braid_pc";
            // ec
            rect_idx["fields/radial_ec/number_of_components"] = 1;
            rect_idx["fields/radial_ec/association"] = "element";
            rect_idx["fields/radial_ec/topology"]    = "mesh";
            rect_idx["fields/radial_ec/path"]        = "rect/fields/radial_ec";


    // structured
        Node &struct_idx = index_root["struct"];
        // state
        struct_idx["state/cycle"] = 42;
        struct_idx["state/time"]  = 3.1415;
        struct_idx["state/number_of_domains"]  = 1;
        // coords
        struct_idx["coordsets/coords/type"]         = "explicit";
        struct_idx["coordsets/coords/coord_system"] = "xy";
        struct_idx["coordsets/coords/path"]         = "struct/coords";
        // topology
        struct_idx["topologies/mesh/type"]     = "structured";
        struct_idx["topologies/mesh/coordset"] = "coords";
        struct_idx["topologies/mesh/path"]     = "struct/topology";
        // fields
            // pc
            struct_idx["fields/braid_pc/number_of_components"] = 1;
            struct_idx["fields/braid_pc/association"] = "point";
            struct_idx["fields/braid_pc/topology"]    = "mesh";
            struct_idx["fields/braid_pc/path"]        = "struct/fields/braid_pc";
            // ec
            struct_idx["fields/radial_ec/number_of_components"] = 1;
            struct_idx["fields/radial_ec/association"] = "element";
            struct_idx["fields/radial_ec/topology"]    = "mesh";
            struct_idx["fields/radial_ec/path"]        = "struct/fields/radial_ec";
    
    // tris (unstructured)
    Node &tris_idx = index_root["tris"];
    // state
    tris_idx["state/cycle"] = 42;
    tris_idx["state/time"]  = 3.1415;
    tris_idx["state/number_of_domains"]  = 1;
    // coords
    tris_idx["coordsets/coords/type"]         = "explicit";
    tris_idx["coordsets/coords/coord_system"] = "xy";
    tris_idx["coordsets/coords/path"]         = "tris/coords";
    // topology
    tris_idx["topologies/mesh/type"]     = "unstructured";
    tris_idx["topologies/mesh/coordset"] = "coords";
    tris_idx["topologies/mesh/path"]     = "tris/topology";
    // fields
        // pc
        tris_idx["fields/braid_pc/association"] = "point";
        tris_idx["fields/braid_pc/topology"]    = "mesh";
        tris_idx["fields/braid_pc/number_of_components"] = 1;
        tris_idx["fields/braid_pc/path"]   = "tris/fields/braid_pc";
        // ec
        tris_idx["fields/radial_ec/number_of_components"] = 1;
        tris_idx["fields/radial_ec/association"] = "element";
        tris_idx["fields/radial_ec/topology"]    = "mesh";
        tris_idx["fields/radial_ec/path"]        = "tris/fields/radial_ec";
    
    // quads (unstructured)
    Node &quads_idx = index_root["quads"];
    // state
    quads_idx["state/cycle"] = 42;
    quads_idx["state/time"]  = 3.1415;
    quads_idx["state/number_of_domains"]  = 1;
    // coords
    quads_idx["coordsets/coords/type"]         = "explicit";
    quads_idx["coordsets/coords/coord_system"] = "xy";
    quads_idx["coordsets/coords/path"]         = "quads/coords";
    // topology
    quads_idx["topologies/mesh/type"]     = "unstructured";
    quads_idx["topologies/mesh/coordset"] = "coords";
    quads_idx["topologies/mesh/path"]     = "quads/topology";
    // fields
        // pc
        quads_idx["fields/braid_pc/number_of_components"] = 1;
        quads_idx["fields/braid_pc/association"] = "point";
        quads_idx["fields/braid_pc/topology"]    = "mesh";
        quads_idx["fields/braid_pc/path"]        = "quads/fields/braid_pc";
        // ec
        quads_idx["fields/radial_ec/number_of_components"] = 1;
        quads_idx["fields/radial_ec/association"] = "element";
        quads_idx["fields/radial_ec/topology"]    = "mesh";
        quads_idx["fields/radial_ec/path"]        = "quads/fields/radial_ec";
    
}

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
        uni_idx["coordsets/coords/path"]         = "uniform/coords";
        // topology
        uni_idx["topologies/mesh/type"]     = "uniform";
        uni_idx["topologies/mesh/coordset"] = "coords";
        uni_idx["topologies/mesh/path"]     = "uniform/topology";
        // fields
            // pc
            uni_idx["fields/braid_pc/number_of_components"] = 1;
            uni_idx["fields/braid_pc/association"] = "point";
            uni_idx["fields/braid_pc/topology"]    = "mesh";
            uni_idx["fields/braid_pc/path"]        = "uniform/fields/braid_pc";
            // ec
            uni_idx["fields/radial_ec/number_of_components"] = 1;
            uni_idx["fields/radial_ec/association"] = "element";
            uni_idx["fields/radial_ec/topology"]    = "mesh";
            uni_idx["fields/radial_ec/path"]        = "uniform/fields/radial_ec";

    // rectilinear
        Node &rect_idx = index_root["rect"];
        // state
        rect_idx["state/cycle"] = 42;
        rect_idx["state/time"]  = 3.1415;
        rect_idx["state/number_of_domains"]  = 1;
        // coords
        rect_idx["coordsets/coords/type"]         = "rectilinear";
        rect_idx["coordsets/coords/coord_system"] = "xyz";
        rect_idx["coordsets/coords/path"]         = "rect/coords";
        // topology
        rect_idx["topologies/mesh/type"]     = "rectilinear";
        rect_idx["topologies/mesh/coordset"] = "coords";
        rect_idx["topologies/mesh/path"]     = "rect/topology";
        // fields
            // pc
            rect_idx["fields/braid_pc/number_of_components"] = 1;
            rect_idx["fields/braid_pc/association"] = "point";
            rect_idx["fields/braid_pc/topology"]    = "mesh";
            rect_idx["fields/braid_pc/path"]        = "rect/fields/braid_pc";
            // ec
            rect_idx["fields/radial_ec/number_of_components"] = 1;
            rect_idx["fields/radial_ec/association"] = "element";
            rect_idx["fields/radial_ec/topology"]    = "mesh";
            rect_idx["fields/radial_ec/path"]        = "rect/fields/radial_ec";


    // structured
        Node &struct_idx = index_root["struct"];
        // state
        struct_idx["state/cycle"] = 42;
        struct_idx["state/time"]  = 3.1415;
        struct_idx["state/number_of_domains"]  = 1;
        // coords
        struct_idx["coordsets/coords/type"]         = "explicit";
        struct_idx["coordsets/coords/coord_system"] = "xyz";
        struct_idx["coordsets/coords/path"]         = "struct/coords";
        // topology
        struct_idx["topologies/mesh/type"]     = "structured";
        struct_idx["topologies/mesh/coordset"] = "coords";
        struct_idx["topologies/mesh/path"]     = "struct/topology";
        // fields
            // pc
            struct_idx["fields/braid_pc/number_of_components"] = 1;
            struct_idx["fields/braid_pc/association"] = "point";
            struct_idx["fields/braid_pc/topology"]    = "mesh";
            struct_idx["fields/braid_pc/path"]        = "struct/fields/braid_pc";
            // ec
            struct_idx["fields/radial_ec/number_of_components"] = 1;
            struct_idx["fields/radial_ec/association"] = "element";
            struct_idx["fields/radial_ec/topology"]    = "mesh";
            struct_idx["fields/radial_ec/path"]        = "struct/fields/radial_ec";
    
    // tets (unstructured)
    Node &tets_idx = index_root["tets"];
    // state
    tets_idx["state/cycle"] = 42;
    tets_idx["state/time"]  = 3.1415;
    tets_idx["state/number_of_domains"]  = 1;
    // coords
    tets_idx["coordsets/coords/type"]         = "explicit";
    tets_idx["coordsets/coords/coord_system"] = "xyz";
    tets_idx["coordsets/coords/path"]         = "tets/coords";
    // topology
    tets_idx["topologies/mesh/type"]     = "unstructured";
    tets_idx["topologies/mesh/coordset"] = "coords";
    tets_idx["topologies/mesh/path"]     = "tets/topology";
    // fields
        // pc
        tets_idx["fields/braid_pc/number_of_components"] = 1;
        tets_idx["fields/braid_pc/association"] = "point";
        tets_idx["fields/braid_pc/topology"]    = "mesh";
        tets_idx["fields/braid_pc/path"]        = "tets/fields/braid_pc";
        // ec
        tets_idx["fields/radial_ec/number_of_components"] = 1;
        tets_idx["fields/radial_ec/association"] = "element";
        tets_idx["fields/radial_ec/topology"]    = "mesh";
        tets_idx["fields/radial_ec/path"]        = "tets/fields/radial_ec";
    
    // hexs (unstructured)
    Node &hexs_idx = index_root["hexs"];
    // state
    hexs_idx["state/cycle"] = 42;
    hexs_idx["state/time"]  = 3.1415;
    hexs_idx["state/number_of_domains"]  = 1;
    // coords
    hexs_idx["coordsets/coords/type"]         = "explicit";
    hexs_idx["coordsets/coords/coord_system"] = "xyz";
    hexs_idx["coordsets/coords/path"]         = "hexs/coords";
    // topology
    hexs_idx["topologies/mesh/type"]     = "unstructured";
    hexs_idx["topologies/mesh/coordset"] = "coords";
    hexs_idx["topologies/mesh/path"]     = "hexs/topology";
    // fields
        // pc
        hexs_idx["fields/braid_pc/number_of_components"] = 1;
        hexs_idx["fields/braid_pc/association"] = "point";
        hexs_idx["fields/braid_pc/topology"]    = "mesh";
        hexs_idx["fields/braid_pc/path"]        = "hexs/fields/braid_pc";
        // ec
        hexs_idx["fields/radial_ec/number_of_components"] = 1;
        hexs_idx["fields/radial_ec/association"] = "element";
        hexs_idx["fields/radial_ec/topology"]    = "mesh";
        hexs_idx["fields/radial_ec/path"]        = "hexs/fields/radial_ec";
    
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
    index_t npts_x = 51;
    index_t npts_y = 51;
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

    blueprint::mesh::examples::braid("points_explicit",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points_explicit"]);

    Node expanded;
    
    NodeIterator itr = dsets.children();
    
    while(itr.has_next())
    {
        Node info;
        Node &mesh = itr.next();
        //mesh.print();
        std::string name = itr.path();
        CONDUIT_INFO("expanding 2d example '" << name << "'");
        blueprint::mesh::expand(mesh,expanded[name],info);

    }
    
    if(silo_enabled)
    {
    
        itr = expanded.children();
    
        while(itr.has_next())
        {
            Node &mesh = itr.next();
            std::string name = itr.path();
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
        
        Node root;
        create_blueprint_index_for_2d_examples(root["blueprint_index"]);
        
        root["protocol/name"] = "conduit_hdf5";
        root["protocol/version"] = "0.1";
        
        root["number_of_files"] = 1;
        root["number_of_trees"] = 1;
        root["file_pattern"] = "braid_2d_examples.hdf5";
        root["tree_pattern"] = "/";
        
        CONDUIT_INFO("Creating ")
        CONDUIT_INFO("Creating: braid_2d_examples.hdf5.blueprint_root")
        relay::io::save(root,"braid_2d_examples.hdf5.blueprint_root","hdf5");
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
    index_t npts_x = 51;
    index_t npts_y = 51;
    index_t npts_z = 51; // 3D examples ...
    
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

    blueprint::mesh::examples::braid("points_explicit",
                                     npts_x,
                                     npts_y,
                                     npts_z,
                                     dsets["points_explicit"]);

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

    Node expanded;
    
    NodeIterator itr = dsets.children();
    
    while(itr.has_next())
    {
        Node info;
        Node &mesh = itr.next();
        //mesh.print();
        std::string name = itr.path();
        std::cout << "expanding 3d example '" << name << "'" << std::endl;
        blueprint::mesh::expand(mesh,expanded[name],info);
    }
    
    if(silo_enabled)
    {
    
        itr = expanded.children();
        
        while(itr.has_next())
        {
            Node &mesh = itr.next();
            //mesh.print();
            std::string name = itr.path();
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
        
        Node root;
        create_blueprint_index_for_3d_examples(root["blueprint_index"]);

        root["protocol/name"]    = "hdf5";
        root["protocol/version"] = "0.1";
        
        root["number_of_files"] = 1;
        root["number_of_trees"] = 1;
        root["file_pattern"] = "braid_3d_examples.hdf5";
        root["tree_pattern"] = "/";

        CONDUIT_INFO("Creating: braid_3d_examples.hdf5.blueprint_root")
        relay::io::save(root,"braid_3d_examples.hdf5.blueprint_root","hdf5");
        CONDUIT_INFO("Creating: braid_3d_examples.hdf5")
        relay::io::save(dsets,"braid_3d_examples.hdf5");
    }
    
}
