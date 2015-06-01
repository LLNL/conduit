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
/// file: conduit_mesh_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_mesh.hpp"
#include "conduit_io.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_mesh_examples, uniform_2d)
{
    Node iocfg;
    io::about(iocfg);

    bool silo_enabled = iocfg["endpoints/conduit_silo"].as_string() == "enabled";
        
    Node uniform;
    mesh::examples::braid("uniform",20,20,0,uniform);
    uniform.print();
    uniform.to_pure_json("braid_uniform_example.json");


    Node rect;
    mesh::examples::braid("rectilinear",20,20,0,rect);
    rect.print();
    rect.to_pure_json("braid_rect_example.json");


    Node tris;
    mesh::examples::braid("tris",20,20,0,tris);
    tris.print();
    tris.to_pure_json("braid_quads_example.json");


    Node quads;
    mesh::examples::braid("quads",20,20,0,quads);
    quads.print();
    quads.to_pure_json("braid_quads_example.json");

    
    if(silo_enabled)
    {
        conduit::io::mesh::save(uniform,"braid_uniform_example.silo:uniform2d");
        conduit::io::mesh::save(rect,"braid_rect_example.silo:rect2d");
        conduit::io::mesh::save(tris,"braid_tris_example.silo:tris");
        conduit::io::mesh::save(quads,"braid_quad_example.silo:quad");
    }
    
}
