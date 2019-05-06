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
/// file: conduit_node_iterator.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_NODE_ITERATOR_HPP
#define CONDUIT_NODE_ITERATOR_HPP

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "conduit_node.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

// forward declare NodeConstIterator so it can be a used as a friend
// to NodeIterator
class NodeConstIterator;

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
    friend class NodeConstIterator;
//-----------------------------------------------------------------------------
/// NodeIterator Construction and Destruction
//-----------------------------------------------------------------------------
    /// Default constructor.
    NodeIterator();
    /// Copy constructor.
    NodeIterator(const NodeIterator &itr);
    
    /// Primary iterator constructor.
    NodeIterator(Node *node,index_t idx=0);
    
    /// Primary iterator constructor.
    /// this will use the pointer to the passed Node ref.
    NodeIterator(Node &node,index_t idx=0);
    
    /// Destructor 
    ~NodeIterator();
 
    /// Assignment operator.
    NodeIterator &operator=(const NodeIterator &itr);
 
//-----------------------------------------------------------------------------
/// Iterator value and property access.
//-----------------------------------------------------------------------------
    std::string name()  const;
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
//-----------------------------------------------------------------------------
// -- end conduit::NodeIterator --
//-----------------------------------------------------------------------------


    
//-----------------------------------------------------------------------------
// -- begin conduit::NodeIterator --
//-----------------------------------------------------------------------------
///
/// class: conduit::NodeConstIterator
///
/// description:
///  General purpose const iterator for Nodes.
///
//-----------------------------------------------------------------------------
class CONDUIT_API NodeConstIterator
{
public:
//-----------------------------------------------------------------------------
//
// -- conduit::NodeConstIterator public members --
//
//-----------------------------------------------------------------------------
    
//-----------------------------------------------------------------------------
/// NodeConstIterator Construction and Destruction
//-----------------------------------------------------------------------------
    /// Default constructor.
    NodeConstIterator();
    /// Copy constructor.
    NodeConstIterator(const NodeConstIterator &itr);
    /// Primary iterator constructor.
    NodeConstIterator(const Node *node,index_t idx=0);
    /// Primary iterator constructor.
    /// this will use the pointer to the passed Node ref.
    NodeConstIterator(const Node &node,index_t idx=0);
    /// Destructor 
    ~NodeConstIterator();

    /// Construct from non const
    NodeConstIterator(const NodeIterator &itr);
 
    /// Assignment operator.
    NodeConstIterator &operator=(const NodeConstIterator &itr);

    /// Assignment operator from non const
    NodeConstIterator &operator=(const NodeIterator &itr);
 
//-----------------------------------------------------------------------------
/// Iterator value and property access.
//-----------------------------------------------------------------------------
    std::string name()  const;
    index_t     index() const;
    const Node &node();
    void        to_front();

//-----------------------------------------------------------------------------
/// Iterator forward control.
//-----------------------------------------------------------------------------
    bool        has_next() const;
    const Node &next();
    const Node &peek_next();

//-----------------------------------------------------------------------------
/// Iterator reverse control.
//-----------------------------------------------------------------------------
    bool        has_previous() const;
    const Node &previous();
    const Node &peek_previous();
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
    const Node  *m_node;
    /// current child index
    index_t      m_index;
    /// total number of children 
    index_t      m_num_children; 
};
//-----------------------------------------------------------------------------
// -- end conduit::NodeIterator --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif

