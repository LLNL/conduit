//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://llnl.github.io/conduit/.
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
/// file: blueprint_mesh.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <string.h>
#include <math.h>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "blueprint.hpp"
#include "blueprint_mesh.hpp"

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin blueprint:: --
//-----------------------------------------------------------------------------
namespace blueprint
{


//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    blueprint::about(n);
    return n.to_json();
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    n.reset();
    n["protocols/mesh"] = "enabled";
    n["protocols/mca"]  = "enabled";
}


//---------------------------------------------------------------------------//
bool
verify(const std::string &protocol,
       Node &n,
       Node &info)
{
    bool res = false;
    info.reset();

    if(protocol == "mesh")
    {
        res = mesh::verify(n,info);
    }
    else if(protocol == "mca")
    {
        res = mca::verify(n,info);
    }
    
    return res;
}

//---------------------------------------------------------------------------//
bool
annotate(const std::string &protocol,
         Node &n,
         Node &info)
{
    bool res = false;
    info.reset();

    if(protocol == "mesh")
    {
        res = mesh::annotate(n,info);
    }
    else if(protocol == "mca")
    {
        res = mca::annotate(n,info);
    }
    
    return res;
}



//---------------------------------------------------------------------------//
bool
transform(const std::string &protocol,
          Node &src,
          Node &actions,
          Node &des,
          Node &info)
{
    bool res = false;
    des.reset();
    info.reset();

    if(protocol == "mesh")
    {
        res = mesh::transform(src,actions,des,info);
    }
    else if(protocol == "mca")
    {
        res = mca::transform(src,actions,des,info);
    }
    
    return res;
}


}
//-----------------------------------------------------------------------------
// -- end blueprint:: --
//-----------------------------------------------------------------------------
