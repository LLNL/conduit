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
#include "conduit_io.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_2d)
{
    Node io_protos;
    io::about(io_protos);

    bool silo_enabled = io_protos["protocols/conduit_silo"].as_string() == "enabled";

    // we are using one node to hold group of example meshes purely out of convenience  
    Node dsets;
    index_t npts_x = 11;
    index_t npts_y = 11;
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
        std::cout << "expanding 2d example '" << name << "'" << std::endl;
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
            std::cout << "saving 2d example '" << name << "' to silo" << std::endl;
            io::save("conduit_silo_mesh",mesh,"braid_2d_" + name +  "_example.silo:mesh");
        }
    }
    
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_examples, mesh_3d)
{
    Node io_protos;
    io::about(io_protos);

    bool silo_enabled = io_protos["protocols/conduit_silo"].as_string() == "enabled";

    // we are using one node to hold group of example meshes purely out of convenience  
    Node dsets;
    index_t npts_x = 11;
    index_t npts_y = 11;
    index_t npts_z = 11; // 3D examples ...
    
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
            std::cout << "saving 3d example '" << name << "' to silo" << std::endl;

            // Skip output of silo mesh for mixed mesh of hexs and tets for now.
            // The silo output is not yet defined and it throws an exception
            // in conduit_silo.cpp::silo_write_ucd_zonelist()
            // in the following line that is looking for the 'shape' node:
            //              std::string topo_shape = shape_block->fetch("shape").as_string();
            // which does not exist for indexed_stream meshes.
            // The silo writer needs to be updated for this case.
            if(name == "hexs_and_tets")
            {
                std::cout<<"\tskipping output to SILO -- this is not implemented yet for indexed_stream meshes."<< std::endl;
                continue;
            }
            io::save("conduit_silo_mesh",mesh,"braid_3d_" + name +  "_example.silo:mesh");
        }
    }
    
}
