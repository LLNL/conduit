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
/// file: conduit_blueprint_mcarray.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mcarray.hpp"
#include "conduit_log.hpp"

//-----------------------------------------------------------------------------
// -- standard cpp lib includes -- 
//-----------------------------------------------------------------------------
#include <algorithm>
#include <map>
#include <set>
#include <limits>

using namespace conduit;
// Easier access to the Conduit logging functions
using namespace conduit::utils;

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
    // mcarray doens't provide any nested protocols

    info.reset();
    log::validation(info,false);
    return false;
}


//----------------------------------------------------------------------------
bool verify(const conduit::Node &n,
            Node &info)
{
    info.reset();
    bool res = true;

    const std::string proto_name = "mcarray";

    // mcarray needs to be an object or a list
    if( ! (n.dtype().is_object() || n.dtype().is_list()) )
    {
        log::error(info,proto_name,"Node has no children");
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

                log::error(info,proto_name,oss.str());

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

            log::error(info,proto_name,oss.str());
            res = false;
        }
    }
    
    log::validation(info,res);

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
        index_t elem_bytes = chld.dtype().element_bytes();
        
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
    index_t stride = 0;
    index_t curr_offset = 0;
   
    while(itr.has_next())
    {
        // get the next child
        const Node &chld = itr.next();
        index_t elem_bytes = DataType::default_dtype(chld.dtype().id()).element_bytes();
        stride += elem_bytes;
    }
    
    itr.to_front();
    
    while(itr.has_next())
    {
        // get the next child
        const Node &chld = itr.next();
        // get the next child's name
        std::string name = itr.name();
        
        // use the child's data type to seed our desired data type
        DataType curr_dt = chld.dtype();

        // get the proper number of bytes for this data type, so we can use it
        // as the stride
        index_t elem_bytes = DataType::default_dtype(curr_dt.id()).element_bytes();
                
        // set the stride and offset
        curr_dt.set_stride(stride);
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

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mlarray --
//-----------------------------------------------------------------------------
namespace mlarray
{

//-----------------------------------------------------------------------------
bool
verify(const std::string &/*protocol*/,
       const Node &/*n*/,
       Node &info)
{
    // mlarray doens't provide any nested protocols
    info.reset();
    log::validation(info,false);
    return false;
}

//----------------------------------------------------------------------------
bool verify(const conduit::Node &n,
            Node &info)
{
    return mlarray::verify(n, info, 0, std::numeric_limits<index_t>::max());
}

//----------------------------------------------------------------------------
bool verify(const conduit::Node &n,
            Node &info,
            const index_t min_depth,
            const index_t max_depth)
{
    info.reset();
    bool res = true;

    const std::string protocol = "mlarray";

    if(n.dtype().is_empty())
    {
        log::error(info,protocol,"mlarray is empty");
        res = false;
    }

    // TODO(JRC): Consider updating the logging in this function to better
    // indicate the exact points in the subtree rooted at 'n' at which
    // there are problems.
    // TODO(JRC): This function currently doesn't support list ML-arrays, but
    // it should at some point in the future.

    // Organize Nodes by Tree Depth //

    typedef std::map<const conduit::Node*, index_t> NodeMap;
    typedef std::vector<const conduit::Node*> NodeVector;

    NodeMap node_depth_map;
    index_t node_max_depth = 0;
    {
        std::vector<const conduit::Node*> n_node_stack( 1, &n );
        std::vector<index_t> n_depth_stack( 1, 0 );
        while(!n_node_stack.empty())
        {
            const conduit::Node *curr_node = n_node_stack.back();
            n_node_stack.pop_back();
            index_t curr_depth = n_depth_stack.back();
            n_depth_stack.pop_back();
            if(node_depth_map.find(curr_node) == node_depth_map.end())
            {
                node_depth_map[curr_node] = curr_depth;
                node_max_depth = std::max(node_max_depth, curr_depth);

                NodeConstIterator child_it = curr_node->children();
                while(child_it.has_next())
                {
                    const Node &curr_child = child_it.next();
                    n_node_stack.push_back(&curr_child);
                    n_depth_stack.push_back(curr_depth + 1);
                }
            }
        }
    }

    std::vector<NodeVector> nodes_by_depth((size_t)(node_max_depth + 1));
    for(NodeMap::iterator node_it = node_depth_map.begin();
        node_it != node_depth_map.end(); ++node_it)
    {
        nodes_by_depth[(size_t)node_it->second].push_back(node_it->first);
    }

    // Verify Uniformity of Internal Tree Structure //

    for(index_t curr_depth = 0; curr_depth < node_max_depth; curr_depth++)
    {
        const NodeVector &depth_nodes = nodes_by_depth[(size_t)curr_depth];
        std::vector<std::string> depth_children = depth_nodes[0]->child_names();
        std::set<std::string> depth_childset(depth_children.begin(), depth_children.end());

        for(index_t node_idx = 0; node_idx < (index_t)depth_nodes.size() && res; node_idx++)
        {
            const conduit::Node* curr_node = depth_nodes[(size_t)node_idx];
            const std::vector<std::string> curr_children = curr_node->child_names();
            std::set<std::string> curr_childset(curr_children.begin(), curr_children.end());

            if(curr_childset != depth_childset)
            {
                std::ostringstream oss;
                oss << "node subdepth " << curr_depth << " has subtree mismatches";
                log::error(info,protocol,oss.str());
                res = false;
            }
        }
    }

    // Verify Correctness/Uniformity of Leaves //

    {
        index_t curr_depth = node_max_depth;
        const NodeVector &depth_nodes = nodes_by_depth[(size_t)curr_depth];
        const index_t depth_elems = depth_nodes[0]->dtype().number_of_elements();

        for(index_t node_idx = 0; node_idx < (index_t)depth_nodes.size() && res; node_idx++)
        {
            const conduit::Node* curr_node = depth_nodes[(size_t)node_idx];

            if(!curr_node->dtype().is_number())
            {
                log::error(info,protocol,"node leaves have non-numerical types");
                res = false;
            }
            else if(curr_node->dtype().number_of_elements() != depth_elems)
            {
                log::error(info,protocol,"node leaves have element count differences");
                res = false;
            }
        }
    }

    // Verify Proper Depth Level //

    if(node_max_depth < min_depth || node_max_depth > max_depth)
    {
        std::ostringstream oss;
        oss << "mlarray depth has depth " << node_max_depth <<
            ", which isn't in the required depth bounds of " <<
            "[" << min_depth << ", " << max_depth << "]";
        log::error(info,protocol,oss.str());
        res = false;
    }

    return res;
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mlarray --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
