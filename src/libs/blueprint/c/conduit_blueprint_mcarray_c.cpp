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
/// file: conduit_blueprint_mcarray_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mcarray.h"

#include "conduit.hpp"
#include "conduit_blueprint.hpp"

#include "conduit_cpp_to_c.hpp"


//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
/// Verify passed node confirms to the blueprint mcarray protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_verify(const conduit_node *cnode,
                                 conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mcarray::verify(n,info);
}


//-----------------------------------------------------------------------------
/// Verify passed node confirms to given blueprint mcarray sub protocol.
//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_verify_sub_protocol(const char *protocol,
                                              const conduit_node *cnode,
                                              conduit_node *cinfo)
{
    const Node &n = cpp_node_ref(cnode);
    Node &info    = cpp_node_ref(cinfo);
    return (int)blueprint::mcarray::verify(std::string(protocol),n,info);
}


//----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_is_interleaved(const conduit_node *cnode)
{
    const Node &n = cpp_node_ref(cnode);
    return (int)blueprint::mcarray::is_interleaved(n);
}

//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_to_contiguous(const conduit_node *cnode,
                                        conduit_node *cdest)
{
    const Node &n = cpp_node_ref(cnode);
    Node &dest    = cpp_node_ref(cdest);
    return (int)blueprint::mcarray::to_contiguous(n,dest);
}

//-----------------------------------------------------------------------------
int
conduit_blueprint_mcarray_to_interleaved(const conduit_node *cnode,
                                         conduit_node *cdest)
{
    const Node &n = cpp_node_ref(cnode);
    Node &dest    = cpp_node_ref(cdest);
    return (int)blueprint::mcarray::to_interleaved(n,dest);
}


//-----------------------------------------------------------------------------
/// Interface to generate example data
//-----------------------------------------------------------------------------
void
conduit_blueprint_mcarray_examples_xyz(const char *mcarray_type,
                                       conduit_index_t npts,
                                       conduit_node *cres)
{
    Node &res = cpp_node_ref(cres);
    blueprint::mcarray::examples::xyz(std::string(mcarray_type),
                                      npts,
                                      res);
}



}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

