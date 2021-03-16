// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_o2mrelation.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_utils.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"
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
// access conduit blueprint mesh utilities
namespace o2mutils = conduit::blueprint::o2mrelation::utils;

typedef conduit::blueprint::o2mrelation::O2MIterator O2MIterator;

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
    bool res = true;

    const std::string proto_name = "o2mrelation";

    if(!n.dtype().is_object())
    {
        log::error(info,proto_name,"base node is not an object");
        res = false;
    }

    // Verify Correctness of Meta Sections //

    std::set<const conduit::Node*> o2m_nodeset;
    for(const std::string &o2m_path : o2mutils::O2M_PATHS)
    {
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
std::vector<std::string> data_paths(const conduit::Node &o2mrelation)
{
    std::vector<std::string> paths;

    NodeConstIterator o2miter = o2mrelation.children();
    while(o2miter.has_next())
    {
        const Node &nchld = o2miter.next();
        const std::string &nchld_name = o2miter.name();
        if(std::find(o2mutils::O2M_PATHS.begin(), o2mutils::O2M_PATHS.end(), nchld_name) == o2mutils::O2M_PATHS.end() &&
            nchld.dtype().is_number())
        {
            paths.push_back(nchld_name);
        }
    }

    return paths;
}


//----------------------------------------------------------------------------
void compact_to(const conduit::Node &o2mrelation,
                conduit::Node &res)
{
    res.reset();

    // NOTE(JRC): Compaction only occurs in the case where sizes/offsets exist
    // because otherwise the data must already be compact due to the default
    // values of sizes and offsets.
    if(!o2mrelation.has_child("sizes"))
    {
        res.set_external(o2mrelation);
    }
    else
    {
        O2MIterator o2miter(o2mrelation);
        const std::vector<std::string> o2m_paths_curr =
            conduit::blueprint::o2mrelation::data_paths(o2mrelation);

        const conduit::Node &o2m_offsets = o2mrelation["offsets"];
        const conduit::Node &o2m_sizes = o2mrelation["sizes"];
        conduit::Node &res_offsets = res["offsets"];
        conduit::Node &res_sizes = res["sizes"];
        const conduit::DataType offsets_dtype(o2m_offsets.dtype().id(), 1);
        const conduit::DataType sizes_dtype(o2m_offsets.dtype().id(), 1);
        {
            res_sizes.set(o2m_sizes);
            res_offsets.set(conduit::DataType(offsets_dtype.id(),
                o2miter.elements(conduit::blueprint::o2mrelation::ONE)));

            for(index_t pi = 0; pi < (index_t)o2m_paths_curr.size(); pi++)
            {
                const std::string& o2m_path = o2m_paths_curr[pi];
                res[o2m_path].set(conduit::DataType(o2mrelation[o2m_path].dtype().id(),
                    o2miter.elements(conduit::blueprint::o2mrelation::DATA)));
            }
        }

        conduit::Node o2m_temp, res_temp;
        index_t curr_index = 0, curr_offset = 0;
        while(o2miter.has_next(conduit::blueprint::o2mrelation::ONE))
        {
            const index_t one_index = o2miter.next(conduit::blueprint::o2mrelation::ONE);

            res_temp.set_external(offsets_dtype, res_offsets.element_ptr(one_index));
            o2m_temp.set(curr_offset);
            o2m_temp.to_data_type(offsets_dtype.id(), res_temp);

            o2miter.to_front(conduit::blueprint::o2mrelation::MANY);
            while(o2miter.has_next(conduit::blueprint::o2mrelation::MANY))
            {
                o2miter.next(conduit::blueprint::o2mrelation::MANY);
                const index_t data_index = o2miter.index(conduit::blueprint::o2mrelation::DATA);

                for(index_t pi = 0; pi < (index_t)o2m_paths_curr.size(); pi++, curr_index++)
                {
                    const conduit::Node &o2m_data = o2mrelation[o2m_paths_curr[pi]];
                    conduit::Node &res_data = res[o2m_paths_curr[pi]];

                    const conduit::DataType data_dtype(o2m_data.dtype().id(), 1);
                    o2m_temp.set_external(data_dtype, (void*)o2m_data.element_ptr(data_index));
                    res_temp.set_external(data_dtype, (void*)res_data.element_ptr(curr_index));
                    o2m_temp.to_data_type(data_dtype.id(), res_temp);
                }
            }

            curr_offset += o2miter.elements(conduit::blueprint::o2mrelation::MANY);
        }
    }
}


//----------------------------------------------------------------------------
bool generate_offsets(conduit::Node &n,
                      conduit::Node &info)
{
    bool res = true;

    const std::string proto_name = "o2mrelation::generate_offsets";

    if(!n.dtype().is_object())
    {
        log::error(info,proto_name,"base node is not an object");
        res = false;
    }
    else if(!n.has_child("sizes"))
    {
        log::error(info,proto_name,"missing 'sizes' child");
        res = false;
    }
    else
    {
        conduit::Node &sizes = n["sizes"];
        conduit::Node &offsets = n["offsets"];

        conduit::Node temp;
        std::vector<int64> offset_array(sizes.dtype().number_of_elements());
        for(index_t o = 0; o < (index_t)offset_array.size(); o++)
        {
            temp.set_external(conduit::DataType(sizes.dtype().id(), 1),
                const_cast<void*>(sizes.element_ptr(o)));
            offset_array[o] = (o > 0) ? offset_array[o-1] + temp.to_int64() : 0;
        }

        offsets.reset();
        temp.set_external(offset_array);
        temp.to_data_type(sizes.dtype().id(), offsets);
    }

    return res;
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
