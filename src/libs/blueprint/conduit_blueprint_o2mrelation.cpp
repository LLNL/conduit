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
/// file: conduit_blueprint_o2mrelation.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_o2mrelation.hpp"
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

static const std::string o2m_path_list[] = {"sizes", "offsets", "indices"};
static const std::vector<std::string> o2m_paths(o2m_path_list,
    o2m_path_list + sizeof(o2m_path_list) / sizeof(o2m_path_list[0]));

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
// -- begin conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------
namespace o2mrelation
{

//-----------------------------------------------------------------------------
bool
verify(const std::string &/*protocol*/,
       const Node &/*n*/,
       Node &info)
{
    // o2mrelation doens't provide any nested protocols

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

    const std::string proto_name = "o2mrelation";

    if(!n.dtype().is_object())
    {
        log::error(info,proto_name,"base node is not an object");
        res = false;
    }

    // Verify Correctness of Meta Sections //

    std::set<const conduit::Node*> o2m_nodeset;
    for(index_t path_idx = 0; path_idx < (index_t)o2m_paths.size(); path_idx++)
    {
        const std::string &o2m_path = o2m_paths[path_idx];
        const conduit::Node *o2m_node = n.fetch_ptr(o2m_path);
        o2m_nodeset.insert(o2m_node);

        if(o2m_node != NULL && !o2m_node->dtype().is_integer())
        {
            std::ostringstream oss;
            oss << "'" << o2m_path << "' metadata uses non-index type";
            log::error(info,proto_name,oss.str());
            res = false;
        }
    }

    const conduit::Node *sizes_node = n.fetch_ptr("sizes");
    const conduit::Node *offsets_node = n.fetch_ptr("offsets");
    // const conduit::Node *indices_node = n.fetch_ptr("indices");

    if(sizes_node != NULL || offsets_node != NULL)
    {
        if(!(sizes_node != NULL && offsets_node != NULL))
        {
            log::error(info,proto_name,"requires both 'sizes' and 'offsets' specs");
            res = false;
        }
        else if(sizes_node->dtype().number_of_elements() !=
                offsets_node->dtype().number_of_elements())
        {
            log::error(info,proto_name,"requires equal length 'sizes' and 'offsets' specs");
            res = false;
        }
    }

    // Verify Correctness of Relation Section(s) //

    std::set<const conduit::Node*> data_nodeset;

    NodeConstIterator niter = n.children();
    while(niter.has_next())
    {
        const Node &nchld = niter.next();
        const std::string &nchld_name = niter.name();
        if(o2m_nodeset.find(&nchld) == o2m_nodeset.end() && nchld.dtype().is_number())
        {
            std::ostringstream oss;
            oss << "applying relation to path '" << nchld_name << "'";
            log::info(info,proto_name,oss.str());
            data_nodeset.insert(&nchld);
        }
    }

    if(data_nodeset.empty())
    {
        log::error(info,proto_name,"need at least one relation data array");
        res = false;
    }

    // NOTE(JRC): Assuming that values in a relation are unique for a one-to-one
    // pair (i.e. no two sources share a target value by having duplicates in the
    // 'indices' array), then checks can be added here to assert that each relation
    // is at least as large as the s/o/i arrays.

    log::validation(info,res);

    return res;
}


//----------------------------------------------------------------------------
void query_paths(const conduit::Node &n, Node &res)
{
    res.reset();

    res.set(DataType::object());

    NodeConstIterator niter = n.children();
    while(niter.has_next())
    {
        const Node &nchld = niter.next();
        const std::string &nchld_name = niter.name();
        if(std::find(o2m_paths.begin(), o2m_paths.end(), nchld_name) == o2m_paths.end() &&
            nchld.dtype().is_number())
        {
            res[nchld_name];
        }
    }
}


//----------------------------------------------------------------------------
void to_compact(conduit::Node &o2mrelation)
{
    // NOTE(JRC): Compaction only occurs in the case where sizes/offsets exist
    // because otherwise the data must already be compact due to the default
    // values of sizes and offsets.
    if(o2mrelation.has_child("sizes"))
    {
        conduit::Node &offsets = o2mrelation["offsets"];
        conduit::Node &sizes = o2mrelation["sizes"];

        std::vector<int64> offset_array(sizes.dtype().number_of_elements());
        for(index_t o = 0; o < (index_t)offset_array.size(); o++)
        {
            const Node size_node(conduit::DataType(sizes.dtype().id(), 1),
                const_cast<void*>(sizes.element_ptr(o)), true);
            offset_array[o] = (o > 0) ? offset_array[o-1] + size_node.to_int64() : 0;
        }

        Node offset_node;
        offset_node.set_external(offset_array);
        offset_node.to_data_type(offsets.dtype().id(), offsets);
    }
}

//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
