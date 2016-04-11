//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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
#include "blueprint_mesh.hpp"

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

namespace mesh
{

//-----------------------------------------------------------------------------
bool
verify(Node &n,
       Node &info)
{
    return false;
}


//-----------------------------------------------------------------------------
bool
transform(Node &src,
          Node &actions,
          Node &dest,
          Node &info)
{
   // TODO: list vs object case?
   // list example:
   //
   // ["expand"]
   // obj example
   // [ {name: expand, opts: ... }]
   //
   // blueprint::actions::expand(actions,adest);
   
   NodeIterator itr = actions.children();
   
   while(itr.has_next())
   {
       Node &curr = itr.next();
       std::string action_name = curr["name"].as_string();
       if( action_name == "expand")
       {
           bool res = expand(src,dest,info.append());
           if(!res)
           {
               return res;
           }
       }
       else
       {
           std::ostringstream oss;
           oss << "blueprint::mesh, unsupported action:" << action_name;
           info.set(oss.str());
           return false;
       }
   }
   
   return true;

}


//---------------------------------------------------------------------------//
bool
expand(Node &src,
       Node &des,
       Node &info)
{
    if(src.has_path("topologies") && src.has_path("coordsets"))
    {
        // assume all is well, we already have a multi-topology description
        des.set_external(src);
    }
    else if(src.has_path("topology") && src.has_path("coords"))
    {
        // promote single topology to standard multi-topology description
        if(src.has_path("state"))
        {
            des["state"].set_external(src["state"]);
            
        }
        // mesh is the default name for a topo, 
        // coords is the default name for a coordset
        des["coordsets/coords"].set_external(src["coords"]);
        des["topologies/mesh"].set_external(src["topology"]);
        des["topologies/mesh/coordset"].set("coords");
        
        if(src.has_path("fields"))
        {
            des["fields"].set_external(src["fields"]);
            
            NodeIterator itr = des["fields"].children();
            
            while( itr.has_next() )
            {
                Node &field = itr.next();
                field["topologies"].append().set("mesh");
            }
        }
    }
    else
    {
        CONDUIT_ERROR("Missing topologies and coordsets, or topology and coords");
    }
    
    return true;
}


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

