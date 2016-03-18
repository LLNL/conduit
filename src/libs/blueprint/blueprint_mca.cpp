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
/// file: blueprint_mca.cpp
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
#include "blueprint_mca.hpp"

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin blueprint:: --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin blueprint::mca --
//-----------------------------------------------------------------------------

namespace mca
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
annotate(Node &n,
         Node &info)
{
    return true;
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
       // TODO: wire in any mca methods
       // if( action_name == "expand")
       // {
       //     bool res = expand(src,dest,info.append());
       //     if(!res)
       //     {
       //         return res;
       //     }
       // }
       // else
       // {
           std::ostringstream oss;
           oss << "blueprint::mca, unsupported action:" << action_name;
           info.set(oss.str());
           return false;
       // }
   }
   
   return true;

}


//-----------------------------------------------------------------------------
bool
to_contiguous(conduit::Node &src,
              conduit::Node &dest)
{
    // goal is to setup dest with children with the same names as src
    // that point into the desired layout
    //index_t num_dest_elems = src.child(0).number_of_elements() * num_comps;
    
    Schema s_dest;
    NodeIterator itr = src.children();
    
    index_t curr_offset = 0;
    
    while(itr.has_next())
    {
        // get the next child
        Node &chld = itr.next();
        // get the next child's name
        std::string name = itr.path();
        
        // use the child's data type to see our desired data type
        DataType curr_dt = chld.dtype();

        // get the proper number of bytes for this data type, so we can use it
        // as the stride
        index_t elem_bytes = DataType::default_dtype(curr_dt.id()).element_bytes();
        
        // set the stride and offset
        curr_dt.set_stride(elem_bytes);
        curr_dt.set_offset(curr_offset);
        
        // put the dtype into our schema with the correct name
        s_dest[name] = curr_dt;
        
        // update the offset for the next component
        curr_offset += elem_bytes * curr_dt.number_of_elements();
    }
    
    // allocate using our schema
    dest.set(s_dest);
    // copy the data from the source
    dest.update(src);
    
    return true; // we always work!
}



//-----------------------------------------------------------------------------
bool
to_interleaved(conduit::Node &src,
               conduit::Node &dest)
{
    // goal is to setup dest with children with the same names as src
    // that point into the desired layout
    
    Schema s_dest;
    
    NodeIterator itr = src.children();
    index_t num_comps = src.number_of_children();
    index_t curr_offset = 0;
    
    while(itr.has_next())
    {
        // get the next child
        Node &chld = itr.next();
        // get the next child's name
        std::string name = itr.path();
        
        // use the child's data type to see our desired data type
        DataType curr_dt = chld.dtype();

        // get the proper number of bytes for this data type, so we can use it
        // as the stride
        index_t elem_bytes = DataType::default_dtype(curr_dt.id()).element_bytes();
        
        // ASSUMES THE elem_bytes is the same for each one of the components
        // eg: all float64
        
        // set the stride and offset
        curr_dt.set_stride(num_comps * elem_bytes);
        curr_dt.set_offset(curr_offset);
        
        // put the dtype into our schema with the correct name
        s_dest[name] = curr_dt;
        
        // update the offset for the next component
        curr_offset += elem_bytes;
    }
    
    // allocate using our schema
    dest.set(s_dest);
    // copy the data from the source
    dest.update(src);
    
    return true; // we always work!
}


//----------------------------------------------------------------------------
bool BLUEPRINT_API is_contiguous(conduit::Node &n)
{
    // TODO: Implement
    
    // Conditions:
    // is_compact()
    // 1) address + offset for each comp can back tracks to start address
    //    (comp address + offset - func(comp index) == start address)
    // 2) is_compact()
    
    
    uint8 *starting_data_ptr = NULL;
    bool ok = true;
    
    index_t total_bytes = 0;
    NodeIterator itr = n.children();
        
    while(itr.has_next() && ok)
    {
        // get the next child
        Node &chld = itr.next();
        
        if(starting_data_ptr == NULL)
        {
           starting_data_ptr = (uint8*) chld.element_ptr(0);
        }
        
        if(chld.is_compact())
        {
            uint8 *current_data_ptr = (uint8*) chld.element_ptr(0);
            ok = ( (current_data_ptr - total_bytes) == starting_data_ptr );
            total_bytes += chld.total_bytes();
        }
        else
        {
            ok = false;
        }
    }
    
    return ok;
}

//----------------------------------------------------------------------------
bool BLUEPRINT_API is_interleaved(conduit::Node &n)
{
    // TODO: Implement

    // Conditions:
    // 1) address + offset for each comp can back tracks to start address
    //    (comp address + offset - func(comp index) == start address)
    // 2) strides are the same as ~ (element bytes * num comps)
    
    return false;
}

};
//-----------------------------------------------------------------------------
// -- end blueprint::mesh --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end blueprint:: --
//-----------------------------------------------------------------------------
