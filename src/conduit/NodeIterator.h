//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: NodeIterator.h
///
//-----------------------------------------------------------------------------

#ifndef __CONDUIT_NODE_ITERATOR_H
#define __CONDUIT_NODE_ITERATOR_H

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "Error.h"
#include "Node.h"
#include "Utils.h"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{
    
//-----------------------------------------------------------------------------
// -- begin conduit::NodeIterator --
//-----------------------------------------------------------------------------
///
/// class: conduit::NodeIterator
///
/// description:
///  General purpose iterator for Nodes.
///
//-----------------------------------------------------------------------------
class CONDUIT_API NodeIterator
{
public:
//-----------------------------------------------------------------------------
//
// -- conduit::NodeIterator public members --
//
//-----------------------------------------------------------------------------
    
//-----------------------------------------------------------------------------
/// NodeIterator Construction and Destruction
//-----------------------------------------------------------------------------
    /// Default constructor.
    NodeIterator();
    /// Copy constructor.
    NodeIterator(const NodeIterator &itr);
    /// Primary iterator constructor.
    NodeIterator(Node *node,index_t idx=0);
    /// Destructor 
    ~NodeIterator();
 
    /// Assignment operator.
    NodeIterator &operator=(const NodeIterator &itr);
 
//-----------------------------------------------------------------------------
/// Iterator value and property access.
//-----------------------------------------------------------------------------
    std::string path()  const;
    index_t     index() const;
    Node       &node();
    void        to_front();

//-----------------------------------------------------------------------------
/// Iterator forward control.
//-----------------------------------------------------------------------------
    bool        has_next() const;
    Node       &next();
    Node       &peek_next();

//-----------------------------------------------------------------------------
/// Iterator reverse control.
//-----------------------------------------------------------------------------
    bool        has_previous() const;
    Node       &previous();
    Node       &peek_previous();
    void        to_back();

//-----------------------------------------------------------------------------
/// Human readable info about this iterator
//-----------------------------------------------------------------------------
    void        info(Node &res) const;
    
private:
//-----------------------------------------------------------------------------
//
// -- conduit::NodeIterator private data members --
//
//-----------------------------------------------------------------------------
    /// pointer to the Node wrapped by this iterator 
    Node    *m_node;
    /// current child index
    index_t  m_index;
    /// total number of children 
    index_t  m_num_children; 
    
};

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif

