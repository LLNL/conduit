//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// file: NodeIterator.cpp
///

#include "NodeIterator.h"

namespace conduit
{

//============================================
// NodeIterator
//============================================

/* Constructors */
//============================================
NodeIterator::NodeIterator()
: m_node(NULL),
  m_index(0),
  m_num_ele(0)
{
    
}

//============================================
NodeIterator::NodeIterator(Node *node,index_t idx)
:m_node(node),
 m_index(idx)
{
    m_num_ele = node->number_of_entries();
}


//============================================
NodeIterator::NodeIterator(const NodeIterator &itr)
:m_node(itr.m_node),
 m_index(itr.m_index),
 m_num_ele(itr.m_index)
{

}

//============================================
NodeIterator::~NodeIterator()
{
    
}
 
 
//============================================
NodeIterator &
NodeIterator::operator=(const NodeIterator &itr)
{
    if(this != &itr)
    {
        m_node    = itr.m_node;
        m_index   = itr.m_index;
        m_num_ele = itr.m_num_ele;
    }
    return *this;
}
 
 
//============================================
std::string
NodeIterator::path() const
{
    return m_node->m_schema->object_order()[m_index-1];
}

//============================================
index_t
NodeIterator::index() const
{
    return m_index-1;
}

//============================================
Node &
NodeIterator::node()
{
    return m_node->fetch(m_index-1);
}

//============================================
void
NodeIterator::to_front() 
{
    m_index = 0;
}


//============================================
bool
NodeIterator::has_next() const
{
    return ( (m_num_ele != 0) &&
             (m_index < m_num_ele) );
}


//============================================
Node &
NodeIterator::next() 
{
    m_index++;
    return m_node->fetch(m_index-1);
}

//============================================
Node &
NodeIterator::peek_next() 
{
    index_t idx = m_index;
    if(has_next())
    {
        idx++;
    }
    return m_node->fetch(idx-1);
}


//============================================
void
NodeIterator::to_back()
{
    m_index = m_num_ele+1;
}


//============================================
bool
NodeIterator::has_previous() const
{
    return ( m_index > 1 );
}


//============================================
Node &
NodeIterator::previous() 
{
    if(has_previous())
    {
        m_index--;
    }
    return m_node->fetch(m_index-1);
}

//============================================
Node &
NodeIterator::peek_previous() 
{
    index_t idx = m_index;
    if(has_previous())
    {
        idx--;
    }
    return m_node->fetch(idx);
}


//============================================
void
NodeIterator::info(Node &res) const
{
    res.reset();
    res["index"] = m_index;
    res["node_ref"] = utils::to_hex_string(m_node);
    res["number_of_elements"] = m_num_ele;
}

}

