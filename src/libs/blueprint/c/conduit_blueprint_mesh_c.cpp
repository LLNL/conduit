//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_blueprint_mesh_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh.h"

#include "conduit.hpp"
#include "conduit_blueprint.hpp"

#include "conduit_cpp_to_c.hpp"


//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
/// Verify passed node confirms to the blueprint mesh protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mesh_verify(const conduit_node *cnode,
                              conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mesh::verify(n,info);
}


//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint mesh sub protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mesh_verify_sub_protocol(const char *protocol,
                                           const conduit_node *cnode,
                                           conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mesh::verify(std::string(protocol),n,info);
}


//-----------------------------------------------------------------------------
/// Generate mesh::index from valid mesh
//-----------------------------------------------------------------------------
void
conduit_blueprint_mesh_generate_index(const conduit_node *cmesh,
                                      const char *ref_path,
                                      conduit_index_t num_domains,
                                      conduit_node *cindex_out)
{
    const Node &mesh = cpp_node_ref(cmesh);
    Node &index_out  = cpp_node_ref(cindex_out);
    blueprint::mesh::generate_index(mesh,
                                    std::string(ref_path),
                                    num_domains,
                                    index_out);
}

//-----------------------------------------------------------------------------
/// Interface to generate example data
//-----------------------------------------------------------------------------
void
conduit_blueprint_mesh_examples_braid(const char *mesh_type,
                                      conduit_index_t nx,
                                      conduit_index_t ny,
                                      conduit_index_t nz,
                                      conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mesh::examples::braid(std::string(mesh_type),
                                     nx,ny,nz,
                                     res);
}



}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

