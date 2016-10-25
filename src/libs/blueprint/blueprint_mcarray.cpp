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
/// file: blueprint_mcarray.cpp
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
#include "blueprint_mcarray.hpp"

using namespace conduit;

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint:: --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mcarray --
//-----------------------------------------------------------------------------
namespace mcarray
{

//-----------------------------------------------------------------------------
bool
verify(const std::string &/*protocol*/,
       const Node &/*n*/,
       Node &info)
{
    info["valid"] = "false";
    // mcarray doens't provide any nested protocols
    return false;
}


//----------------------------------------------------------------------------
bool verify(const conduit::Node &n,
            Node &info)
{
    bool res = true;

    // mcarray needs to be an object or a list
    if( ! (n.dtype().is_object() || n.dtype().is_list()) )
    {
        info["errors"].append().set("mcarray has no children");
        res = false;
    }

    NodeConstIterator itr = n.children();
    index_t num_elems = 0;

    while(itr.has_next())
    {
        // get the next child
        const Node &chld = itr.next();
        
        // make sure we have a number
        if(chld.dtype().is_number())
        {
            if(num_elems == 0)
            {
                num_elems = chld.dtype().number_of_elements();
            }

            if(chld.dtype().number_of_elements() != num_elems)
            {
                std::ostringstream oss;
                std::string chld_name = itr.name();
                
                if(chld_name.size() == 0)
                {
                    oss << "child [" << itr.index() <<  "]";
                }
                else
                {
                    oss << "child \"" << chld_name << "\"";
                }

                oss << " does not have the same number of "
                    << "elements as mcarray components.";

                info["errors"].append().set(oss.str());

                res = false;
            }
        }
        else
        {
            std::ostringstream oss;
            std::string chld_name = itr.name();

            if(chld_name.size() == 0)
            {
                oss << "child [" << itr.index() <<  "]";
            }
            else
            {
                oss << "child \"" << chld_name << "\"";
            }

            oss << " is not a numeric type.";
            info["errors"].append().set(oss.str());

            res = false;
        }
    }

    return res;
}


//-----------------------------------------------------------------------------
bool
to_contiguous(const conduit::Node &src,
              conduit::Node &dest)
{
    // goal is to setup dest with children with the same names as src
    // that point into the desired layout
    //index_t num_dest_elems = src.child(0).number_of_elements() * num_comps;
    
    Schema s_dest;
    NodeConstIterator itr = src.children();
    
    index_t curr_offset = 0;
    
    while(itr.has_next())
    {
        // get the next child
        const Node &chld = itr.next();
        // get the next child's name
        std::string name = itr.name();
        
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
to_interleaved(const conduit::Node &src,
               conduit::Node &dest)
{
    // goal is to setup dest with children with the same names as src
    // that point into the desired layout
    
    Schema s_dest;
    
    NodeConstIterator itr = src.children();
    index_t num_comps = src.number_of_children();
    index_t curr_offset = 0;
    
    while(itr.has_next())
    {
        // get the next child
        const Node &chld = itr.next();
        // get the next child's name
        std::string name = itr.name();
        
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
bool is_interleaved(const conduit::Node &n)
{
    // TODO: Implement

    // Conditions:
    // 1) address + offset for each comp can back tracks to start address
    //    (comp address + offset - func(comp index) == start address)
    // 2) strides are the same 
    bool ok = true;

    uint8 *starting_data_ptr = NULL;

    NodeConstIterator itr = n.children();
    index_t stride = 0;
    index_t total_bytes_per_tuple = 0;
    
    while(itr.has_next() && ok)
    {
      const Node &child = itr.next();
      if(starting_data_ptr == NULL)
      {
        starting_data_ptr = (uint8*) child.element_ptr(0);
        stride = child.dtype().stride();
      }
      
      //std::cout<<"Pointer "<<(uint64*)child.element_ptr(0)<<std::endl;
      ok = (total_bytes_per_tuple == ((uint8*)child.element_ptr(0) - starting_data_ptr));
      if(ok) ok = (stride == child.dtype().stride());
      total_bytes_per_tuple += child.dtype().element_bytes(); 
    }
    return ok; 
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mcarray --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
