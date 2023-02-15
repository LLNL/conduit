// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_o2mrelation_index.cpp
///
//-----------------------------------------------------------------------------
#include <sstream>

#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_index.hpp"

#include "conduit_error.hpp"
#include "conduit_utils.hpp"

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
// -- begin conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------
namespace o2mrelation
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// O2MIndex
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// O2MIndex Construction and Destruction
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
O2MIndex::O2MIndex(const Node *node)
: m_sizes_node(NULL),
  m_indices_node(NULL),
  m_offsets_node(NULL)
{
    std::vector<std::string> paths = conduit::blueprint::o2mrelation::data_paths(*node);

    if(node->has_child("sizes"))
        m_sizes_node = node->fetch_ptr("sizes");
    if(node->has_child("indices"))
        m_indices_node = node->fetch_ptr("indices");
    if(node->has_child("offsets"))
        m_offsets_node = node->fetch_ptr("offsets");
}

//---------------------------------------------------------------------------//
O2MIndex::O2MIndex(const Node &node)
: O2MIndex(&node)
{
    
}


//---------------------------------------------------------------------------//
O2MIndex::O2MIndex(const O2MIndex &itr)
: m_sizes_node(itr.m_sizes_node),
  m_indices_node(itr.m_indices_node),
  m_offsets_node(itr.m_offsets_node)
{
    
}

//---------------------------------------------------------------------------//
O2MIndex::~O2MIndex()
{
    
}

//---------------------------------------------------------------------------//
O2MIndex &
O2MIndex::operator=(const O2MIndex &itr)
{
    if(this != &itr)
    {
        m_sizes_node = itr.m_sizes_node;
        m_indices_node = itr.m_indices_node;
        m_offsets_node = itr.m_offsets_node;
    }
    return *this;
}

//-----------------------------------------------------------------------------
/// Human readable info about this iterator
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
O2MIndex::info(Node &res) const
{
    res.reset();
    if (m_offsets_node)
    {
        res["offsets"] = *m_offsets_node;
    }
    if (m_sizes_node)
    {
        res["sizes"] = *m_sizes_node;
    }
    if (m_indices_node)
    {
        res["indices"] = *m_indices_node;
    }
}

//-----------------------------------------------------------------------------
/// Get info on the "ones" of this index.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t
O2MIndex::index(index_t one_index, index_t many_index) const
{
    index_t index = 0;

    index_t offset = one_index;
    if(m_offsets_node)
    {
        offset = m_offsets_node->as_index_t_accessor()[one_index];
    }

    index = offset;
    if(m_indices_node)
    {
        index = m_indices_node->as_index_t_accessor()[offset + many_index];
    }
    else
    {
        index += many_index;
    }

    return index;
}


//---------------------------------------------------------------------------//
index_t
O2MIndex::size(index_t one_index) const
{
    index_t nelements = 0;

    if(one_index == -1)
    {
        if(m_offsets_node)
        {
            nelements = m_offsets_node->dtype().number_of_elements();
        }
        else if (m_indices_node)
        {
            nelements = m_indices_node->dtype().number_of_elements();
        }
    }
    else
    {
        if(m_sizes_node)
        {
            nelements = m_sizes_node->as_index_t_accessor()[one_index];
        }
        else
        {
            nelements = 1;
        }
    }

    return nelements;
}

//---------------------------------------------------------------------------//
index_t
O2MIndex::offset(index_t one_index) const
{
    index_t offset = one_index;
    if (m_offsets_node)
    {
        offset = m_offsets_node->as_index_t_accessor()[one_index];
    }
    return offset;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// End O2MIndex
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::o2mrelation --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


