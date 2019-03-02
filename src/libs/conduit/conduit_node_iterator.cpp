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
/// file: conduit_node_iterator.cpp
///
//-----------------------------------------------------------------------------
#include <sstream>

#include "conduit_node_iterator.hpp"
#include "conduit_error.hpp"
#include "conduit_utils.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// NodeIterator
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// NodeIterator Construction and Destruction
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
NodeIterator::NodeIterator()
: m_node(NULL),
  m_index(0),
  m_num_children(0)
{
    
}

//---------------------------------------------------------------------------//
NodeIterator::NodeIterator(Node *node,
                           index_t idx)
:m_node(node),
 m_index(idx)
{
    m_num_children = node->number_of_children();
}

//---------------------------------------------------------------------------//
NodeIterator::NodeIterator(Node &node,
                           index_t idx)
:m_node(&node),
 m_index(idx)
{
    m_num_children = node.number_of_children();
}


//---------------------------------------------------------------------------//
NodeIterator::NodeIterator(const NodeIterator &itr)
:m_node(itr.m_node),
 m_index(itr.m_index),
 m_num_children(itr.m_num_children)
{

}

//---------------------------------------------------------------------------//
NodeIterator::~NodeIterator()
{
    
}
 
 
//---------------------------------------------------------------------------//
NodeIterator &
NodeIterator::operator=(const NodeIterator &itr)
{
    if(this != &itr)
    {
        m_node    = itr.m_node;
        m_index   = itr.m_index;
        m_num_children = itr.m_num_children;
    }
    return *this;
}

//-----------------------------------------------------------------------------
/// Iterator value and property access.
//-----------------------------------------------------------------------------
 
//---------------------------------------------------------------------------//
std::string
NodeIterator::name() const
{
    std::ostringstream oss;

    index_t index = m_index-1;
    if(m_node->m_schema->dtype().is_list())
    {
        oss << index;
    }
    else
    {
        oss << m_node->m_schema->object_order()[(size_t)(index)];
    }

    return oss.str();
}

//---------------------------------------------------------------------------//
index_t
NodeIterator::index() const
{
    return m_index-1;
}

//---------------------------------------------------------------------------//
Node &
NodeIterator::node()
{
    return m_node->child(m_index-1);
}


//-----------------------------------------------------------------------------
/// Iterator forward control.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
NodeIterator::to_front() 
{
    m_index = 0;
}


//---------------------------------------------------------------------------//
bool
NodeIterator::has_next() const
{
    return ( (m_num_children != 0) &&
             (m_index < m_num_children) );
}


//---------------------------------------------------------------------------//
Node &
NodeIterator::next() 
{
    if(has_next())
    {
        m_index++;
    }
    else
    {
        CONDUIT_ERROR("next() when has_next() == false");
    }
    return m_node->child(m_index-1);
}


//---------------------------------------------------------------------------//
Node &
NodeIterator::peek_next() 
{
    index_t idx = m_index;
    if(has_next())
    {
        idx++;
    }
    else
    {
        CONDUIT_ERROR("peek_next() when has_next() == false");
    }
    return m_node->child(idx-1);
}


//---------------------------------------------------------------------------//
void
NodeIterator::to_back()
{
    m_index = m_num_children;
}

//-----------------------------------------------------------------------------
/// Iterator reverse control.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
bool
NodeIterator::has_previous() const
{
    return ( m_index > 1 );
}


//---------------------------------------------------------------------------//
Node &
NodeIterator::previous() 
{
    if(has_previous())
    {
        m_index--;
    }
    else
    {
        CONDUIT_ERROR("previous() when has_previous() == false");
    }
    
    return m_node->child(m_index-1);
}

//---------------------------------------------------------------------------//
Node &
NodeIterator::peek_previous() 
{
    index_t idx = m_index;
    if(has_previous())
    {
        idx--;
    }
    else
    {
        CONDUIT_ERROR("peek_previous() when has_previous() == false");
    }
    return m_node->child(idx);
}

//-----------------------------------------------------------------------------
/// Human readable info about this iterator
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
NodeIterator::info(Node &res) const
{
    res.reset();
    res["index"] = m_index;
    res["node_ref"] = utils::to_hex_string(m_node);
    res["number_of_children"] = m_num_children;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// End NodeIterator
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// NodeConstIterator
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// NodeConstIterator Construction and Destruction
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
NodeConstIterator::NodeConstIterator()
: m_node(NULL),
  m_index(0),
  m_num_children(0)
{

}

//---------------------------------------------------------------------------//
NodeConstIterator::NodeConstIterator(const Node *node,
                                     index_t idx)
:m_node(node),
 m_index(idx)
{
    m_num_children = node->number_of_children();
}


//---------------------------------------------------------------------------//
NodeConstIterator::NodeConstIterator(const Node &node,
                                     index_t idx)
:m_node(&node),
 m_index(idx)
{
    m_num_children = node.number_of_children();
}

//---------------------------------------------------------------------------//
NodeConstIterator::NodeConstIterator(const NodeConstIterator &itr)
:m_node(itr.m_node),
 m_index(itr.m_index),
 m_num_children(itr.m_num_children)
{

}

//---------------------------------------------------------------------------//
NodeConstIterator::~NodeConstIterator()
{
    
}


//---------------------------------------------------------------------------//
NodeConstIterator::NodeConstIterator(const NodeIterator &itr)
:m_node(itr.m_node),
 m_index(itr.m_index),
 m_num_children(itr.m_num_children)
{

}

//---------------------------------------------------------------------------//
NodeConstIterator &
NodeConstIterator::operator=(const NodeConstIterator &itr)
{
    if(this != &itr)
    {
        m_node    = itr.m_node;
        m_index   = itr.m_index;
        m_num_children = itr.m_num_children;
    }
    return *this;
}

//---------------------------------------------------------------------------//
NodeConstIterator &
NodeConstIterator::operator=(const NodeIterator &itr)
{
    m_node    = itr.m_node;
    m_index   = itr.m_index;
    m_num_children = itr.m_num_children;
    return *this;
}

//-----------------------------------------------------------------------------
/// Iterator value and property access.
//-----------------------------------------------------------------------------
 
//---------------------------------------------------------------------------//
std::string
NodeConstIterator::name() const
{
    std::ostringstream oss;

    index_t index = m_index-1;
    if(m_node->m_schema->dtype().is_list())
    {
        oss << index;
    }
    else
    {
        oss << m_node->m_schema->object_order()[(size_t)(index)];
    }

    return oss.str();
}

//---------------------------------------------------------------------------//
index_t
NodeConstIterator::index() const
{
    return m_index-1;
}

//---------------------------------------------------------------------------//
const Node &
NodeConstIterator::node()
{
    return m_node->child(m_index-1);
}


//-----------------------------------------------------------------------------
/// Iterator forward control.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
NodeConstIterator::to_front() 
{
    m_index = 0;
}


//---------------------------------------------------------------------------//
bool
NodeConstIterator::has_next() const
{
    return ( (m_num_children != 0) &&
             (m_index < m_num_children) );
}


//---------------------------------------------------------------------------//
const Node &
NodeConstIterator::next() 
{
    if(has_next())
    {
        m_index++;
    }
    else
    {
        CONDUIT_ERROR("next() when has_next() == false");
    }
    return m_node->child(m_index-1);
}


//---------------------------------------------------------------------------//
const Node &
NodeConstIterator::peek_next() 
{
    index_t idx = m_index;
    if(has_next())
    {
        idx++;
    }
    else
    {
        CONDUIT_ERROR("peek_next() when has_next() == false");
    }
    return m_node->child(idx-1);
}


//---------------------------------------------------------------------------//
void
NodeConstIterator::to_back()
{
    m_index = m_num_children;
}

//-----------------------------------------------------------------------------
/// Iterator reverse control.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
bool
NodeConstIterator::has_previous() const
{
    return ( m_index > 1 );
}


//---------------------------------------------------------------------------//
const Node &
NodeConstIterator::previous() 
{
    if(has_previous())
    {
        m_index--;
    }
    else
    {
        CONDUIT_ERROR("previous() when has_previous() == false");
    }
    
    return m_node->child(m_index-1);
}

//---------------------------------------------------------------------------//
const Node &
NodeConstIterator::peek_previous() 
{
    index_t idx = m_index;
    if(has_previous())
    {
        idx--;
    }
    else
    {
        CONDUIT_ERROR("peek_previous() when has_previous() == false");
    }
    return m_node->child(idx);
}

//-----------------------------------------------------------------------------
/// Human readable info about this iterator
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
NodeConstIterator::info(Node &res) const
{
    res.reset();
    res["index"] = m_index;
    res["node_ref"] = utils::to_hex_string(m_node);
    res["number_of_children"] = m_num_children;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// End NodeConstIterator
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------



}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


