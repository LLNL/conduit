// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_node_iterator.cpp
///
//-----------------------------------------------------------------------------
#include <sstream>

#include "conduit_blueprint_o2mrelation.hpp"
#include "conduit_blueprint_o2mrelation_iterator.hpp"

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
// O2MIterator
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// O2MIterator Construction and Destruction
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
O2MIterator::O2MIterator()
: m_node(NULL),
  m_data_node(NULL),
  m_one_index(0),
  m_many_index(0)
{
    
}

//---------------------------------------------------------------------------//
O2MIterator::O2MIterator(const Node *node)
: m_node(node),
  m_one_index(0),
  m_many_index(0)
{
    std::vector<std::string> paths = conduit::blueprint::o2mrelation::data_paths(*node);
    m_data_node = &node->fetch_existing(paths.front());
}

//---------------------------------------------------------------------------//
O2MIterator::O2MIterator(const Node &node)
: O2MIterator(&node)
{
    
}


//---------------------------------------------------------------------------//
O2MIterator::O2MIterator(const O2MIterator &itr)
: m_node(itr.m_node),
  m_data_node(itr.m_data_node),
  m_one_index(itr.m_one_index),
  m_many_index(itr.m_many_index)
{
    
}

//---------------------------------------------------------------------------//
O2MIterator::~O2MIterator()
{
    
}

//---------------------------------------------------------------------------//
O2MIterator &
O2MIterator::operator=(const O2MIterator &itr)
{
    if(this != &itr)
    {
        m_node = itr.m_node;
        m_data_node = itr.m_data_node;
        m_one_index = itr.m_one_index;
        m_many_index = itr.m_many_index;
    }
    return *this;
}

//-----------------------------------------------------------------------------
/// Iterator value and property access.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t
O2MIterator::index(IndexType itype) const
{
    return index(m_one_index, m_many_index, itype);
}


//---------------------------------------------------------------------------//
index_t
O2MIterator::elements(IndexType itype) const
{
    return elements(m_one_index, itype);
}

//-----------------------------------------------------------------------------
/// Iterator forward control.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
bool
O2MIterator::has_next(IndexType itype) const
{
    bool is_next = false;

    if(itype == DATA)
    {
        is_next = has_next(ONE) || has_next(MANY);
    }
    else if(itype == ONE)
    {
        is_next = m_one_index < elements(0, ONE) - 1;
    }
    else // if(itype == MANY)
    {
        is_next = m_many_index < elements(m_one_index, MANY);
    }

    return is_next;
}


//---------------------------------------------------------------------------//
index_t
O2MIterator::next(IndexType itype)
{
    index_t nindex = 0;

    if(itype == DATA)
    {
        if(m_many_index < elements(m_one_index, MANY))
        {
            m_many_index++;
        }
        else
        {
            m_many_index = 1;
            m_one_index++;
        }

        nindex = index(m_one_index, m_many_index, DATA);
    }
    else if(itype == ONE)
    {
        if(m_many_index < 1)
        {
            m_many_index++;
            nindex = m_one_index;
        }
        else
        {
            nindex = ++m_one_index;
        }
    }
    else // if(itype == MANY)
    {
        nindex = m_many_index++;
    }

    return nindex;
}


//---------------------------------------------------------------------------//
index_t
O2MIterator::peek_next(IndexType itype) const
{
    index_t nindex = 0;

    if(itype == DATA)
    {
        if(m_many_index < elements(m_one_index, MANY))
        {
            nindex = index(m_one_index, m_many_index + 1, DATA);
        }
        else
        {
            nindex = index(m_one_index + 1, 1, DATA);
        }
    }
    else if(itype == ONE)
    {
        nindex = m_one_index + ((m_many_index < 1) ? 0 : 1);
    }
    else // if(itype == MANY)
    {
        nindex = m_many_index;
    }

    return nindex;
}


//---------------------------------------------------------------------------//
void
O2MIterator::to_front(IndexType itype)
{
    if(itype == DATA)
    {
        m_one_index = 0;
        m_many_index = 0;
    }
    else if(itype == ONE)
    {
        m_one_index = 0;
    }
    else // if(itype == MANY)
    {
        m_many_index = 0;
    }
}

//-----------------------------------------------------------------------------
/// Iterator reverse control.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
bool
O2MIterator::has_previous(IndexType itype) const
{
    bool is_prev = false;

    if(itype == DATA)
    {
        is_prev = has_previous(ONE) || has_previous(MANY);
    }
    else if(itype == ONE)
    {
        is_prev = m_one_index > 0;
    }
    else // if(itype == MANY)
    {
        is_prev = m_many_index > 1;
    }

    return is_prev;
}


//---------------------------------------------------------------------------//
index_t
O2MIterator::previous(IndexType itype)
{
    index_t pindex = 0;

    if(itype == DATA)
    {
        if(m_many_index > 1)
        {
            m_many_index--;
        }
        else
        {
            m_many_index = elements(m_one_index - 1, MANY);
            m_one_index--;
        }

        pindex = index(m_one_index, m_many_index, DATA);
    }
    else if(itype == ONE)
    {
        pindex = --m_one_index;
    }
    else // if(itype == MANY)
    {
        pindex = --m_many_index - 1;
    }

    return pindex;
}

//---------------------------------------------------------------------------//
index_t
O2MIterator::peek_previous(IndexType itype) const
{
    index_t pindex = 0;

    if(itype == DATA)
    {
        if(m_many_index > 1)
        {
            pindex = index(m_one_index, m_many_index - 1, DATA);
        }
        else
        {
            pindex = index(m_one_index - 1, elements(m_one_index - 1, MANY), DATA);
        }
    }
    else if(itype == ONE)
    {
        pindex = m_one_index - 1;
    }
    else // if(itype == MANY)
    {
        pindex = m_many_index - 2;
    }

    return pindex;
}


//---------------------------------------------------------------------------//
void
O2MIterator::to_back(IndexType itype)
{
    if(itype == DATA)
    {
        m_one_index = elements(0, ONE);
        m_many_index = 1;
    }
    else if(itype == ONE)
    {
        m_one_index = elements(0, ONE);
    }
    else // if(itype == MANY)
    {
        m_many_index = elements(m_one_index, MANY);
    }
}

//-----------------------------------------------------------------------------
/// Human readable info about this iterator
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
O2MIterator::info(Node &res) const
{
    res.reset();
    res["o2m_ref"] = utils::to_hex_string(m_node);
    res["data_ref"] = utils::to_hex_string(m_data_node);
    res["one_index"] = m_one_index;
    res["many_index"] = m_many_index - 1;
}

//-----------------------------------------------------------------------------
/// Private helper functions
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t
O2MIterator::index(index_t one_index, index_t many_index, IndexType itype) const
{
    index_t index = 0;

    if(itype == DATA)
    {
        index_t offset = one_index;
        if(m_node->has_child("offsets"))
        {
            const conduit::Node &offsets_node = m_node->fetch_existing("offsets");
            const conduit::Node offset_node(
                conduit::DataType(offsets_node.dtype().id(), 1),
                (void*)offsets_node.element_ptr(one_index), true);
            offset = offset_node.to_index_t();
        }

        index = offset;
        if(m_node->has_child("indices"))
        {
            const conduit::Node &indices_node = m_node->fetch_existing("indices");
            const conduit::Node index_node(
                conduit::DataType(indices_node.dtype().id(), 1),
                (void*)indices_node.element_ptr(offset), true);
            index = index_node.to_index_t();
        }

        index += (many_index - 1);
    }
    else if(itype == ONE)
    {
        index = one_index;
    }
    else // if(itype == MANY)
    {
        index = (many_index - 1);
    }

    return index;
}


//---------------------------------------------------------------------------//
index_t
O2MIterator::elements(index_t one_index, IndexType itype) const
{
    index_t nelements = 0;

    if(itype == DATA)
    {
        for(index_t oi = 0; oi < elements(0, ONE); oi++)
        {
            nelements += elements(oi, MANY);
        }
    }
    else if(itype == ONE)
    {
        if(m_node->has_child("sizes"))
        {
            const conduit::Node &sizes_node = m_node->fetch_existing("sizes");
            nelements = sizes_node.dtype().number_of_elements();
        }
        else if(m_node->has_child("indices"))
        {
            const conduit::Node &indices_node = m_node->fetch_existing("indices");
            nelements = indices_node.dtype().number_of_elements();
        }
        else
        {
            nelements = m_data_node->dtype().number_of_elements();
        }
    }
    else // if(itype == MANY)
    {
        // if the one index is too high, we return 0
        if(one_index < elements(0, ONE))
        {
            if(m_node->has_child("sizes"))
            {
                const conduit::Node &sizes_node = m_node->fetch_existing("sizes");
                const conduit::Node size_node(
                    conduit::DataType(sizes_node.dtype().id(), 1),
                    (void*)sizes_node.element_ptr(one_index), true);
                nelements = size_node.to_index_t();
            }
            else
            {
                nelements = 1;
            }
        }
        else
        {
            nelements = 0;
        }
    }

    return nelements;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// End O2MIterator
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


