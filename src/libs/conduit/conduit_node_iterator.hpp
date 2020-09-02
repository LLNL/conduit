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
#include "conduit_core.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

class Node;

/**
 * Class template for defining C++-style iterators for iterating over children
 * of a Node.
 *
 * The class meets the requirements for a random access iterator.
 *
 * @tparam Reference the reference type
 * @tparam Pointer the pointer type
 * @tparam the type of the sub class to use for the CRTP pattern
 */
template<typename Reference, typename Pointer, typename Iter>
class CONDUIT_API NodeChildIteratorBase {
public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef Node value_type;
    typedef Pointer pointer;
    typedef Reference reference;
    typedef index_t difference_type;

    NodeChildIteratorBase() : m_parent(NULL), m_index(0) {}

    NodeChildIteratorBase(NodeChildIteratorBase const &rhs) :
            m_parent(rhs.m_parent), m_index(rhs.m_index) {}

    /**
     * Create a new iterator pointing at the given parent node and index.
     *
     * @param parent the parent node whose children to iterate over
     * @param index the index of the child to which this iterator points
     */
    NodeChildIteratorBase(pointer parent, index_t index=0) :
            m_parent(parent), m_index(index) {}

//#if __cplusplus >= 201103L
//    // If someone has a nice, simple way to accomplish this with C++03,
//    // please change this and enable the test.
//    template <bool is_const_ = is_const, typename = typename std::enable_if<is_const_>::type>
//    node_iterator_template(node_iterator_template<false> &rhs) :
//            m_parent(rhs.get_parent()), m_index(rhs.get_index()) {}
//#endif

    bool operator==(NodeChildIteratorBase const &rhs) const {
        return m_parent == rhs.m_parent && m_index == rhs.m_index;
    }

    bool operator!=(NodeChildIteratorBase const &rhs) const {
        return ! (*this == rhs);
    }

    Iter &operator++() {
        ++m_index;
        return static_cast<Iter&>(*this);
    }

    Iter operator++(int) {
        Iter result = static_cast<Iter&>(*this);
        ++*this;
        return result;
    }

    Iter &operator--() {
        --m_index;
        return static_cast<Iter&>(*this);
    }

    Iter operator--(int) {
        Iter result = static_cast<Iter&>(*this);
        --*this;
        return result;
    }

    reference operator*() {
        return (*this)[0];
    }

    pointer operator->() {
        return m_parent->child_ptr(m_index);
    }

    Iter& operator+=(index_t offset) {
        m_index += offset;
        return static_cast<Iter&>(*this);
    }

    Iter& operator-=(index_t offset) {
        m_index -= offset;
        return static_cast<Iter&>(*this);
    }

    Iter operator+(index_t rhs) const {
        Iter result(static_cast<Iter const &>(*this));
        result += rhs;
        return result;
    }

    Iter operator-(index_t rhs) const {
        Iter result(static_cast<Iter const &>(*this));
        result -= rhs;
        return result;
    }

    index_t operator-(Iter const &rhs) const {
        return m_index - rhs.m_index;
    }

    reference operator[](index_t offset) const {
        return m_parent->child(m_index + offset);
    }

    bool operator<(Iter const &rhs) const {
        return m_index < rhs.m_index;
    }

    bool operator<=(Iter const &rhs) const {
        return m_index <= rhs.m_index;
    }

    bool operator>(Iter const &rhs) const {
        return m_index > rhs.m_index;
    }

    bool operator>=(Iter const &rhs) const {
        return m_index >= rhs.m_index;
    }

    /**
     * Get the node being iterated over (the parent).
     *
     * @return the parent node
     */
    Node const* get_parent() const {
        return m_parent;
    }

    /**
     * Get the index of this iterator's node in the parent node
     *
     * @return the index corresponding to this iterator's position in the
     * parent
     */
    index_t get_index() const {
        return m_index;
    }

    /**
     * Get the name of the node pointed at by this iterator
     * @return the node's name
     */
    std::string name() const;

private:
    pointer m_parent;
    index_t m_index;
};

/**
 * Add an offset to an iterator when the iterator is on the right-hand side.
 *
 * @tparam NodeIter the type of the iterator
 *
 * @param lhs the offset to add to the iterator
 * @param rhs the iterator to which to add the offset
 * @return the offset iterator
 */
template<typename NodeIter>
NodeIter operator+(index_t lhs, NodeIter const &rhs) {
    return rhs + lhs;
}

class NodeChildIterator : public NodeChildIteratorBase<Node&, Node*, NodeChildIterator> {
public:
    NodeChildIterator() : NodeChildIteratorBase<Node&, Node*, NodeChildIterator>() {}

    /**
     * Create a new iterator pointing at the given parent node and index.
     *
     * @param parent the parent node whose children to iterate over
     * @param index the index of the child to which this iterator points
     */
    explicit NodeChildIterator(pointer parent, index_t index=0) :
            NodeChildIteratorBase<Node&, Node*, NodeChildIterator>(parent, index) {}
};

class NodeConstChildIterator :
        public NodeChildIteratorBase<Node const &, Node const *, NodeConstChildIterator> {
public:
    NodeConstChildIterator() :
            NodeChildIteratorBase<Node const &, Node const *, NodeConstChildIterator>() {}

    NodeConstChildIterator(NodeConstChildIterator const &rhs) :
            NodeChildIteratorBase<Node const &, Node const *, NodeConstChildIterator>(rhs) {}

    NodeConstChildIterator(NodeChildIterator const &rhs) :
            NodeChildIteratorBase<Node const &, Node const *, NodeConstChildIterator>(
                    rhs.get_parent(), rhs.get_index()) {}

    /**
     * Create a new iterator pointing at the given parent node and index.
     *
     * @param parent the parent node whose children to iterate over
     * @param index the index of the child to which this iterator points
     */
    explicit NodeConstChildIterator(pointer parent, index_t index=0) :
            NodeChildIteratorBase<Node const &, Node const *, NodeConstChildIterator>(parent, index) {}
};

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

//-----------------------------------------------------------------------------
/// Iterator forward control.
//-----------------------------------------------------------------------------
    bool        has_next() const;
    Node       &next();
    Node       &peek_next();
    void        to_front();

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

//-----------------------------------------------------------------------------
/// Support C++-style iterators
//-----------------------------------------------------------------------------

    NodeChildIterator begin();
    NodeChildIterator end();

    NodeConstChildIterator begin() const {
        return cbegin();
    }
    NodeConstChildIterator end() const {
        return cend();
    }

    NodeConstChildIterator cbegin() const;
    NodeConstChildIterator cend() const;

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


//-----------------------------------------------------------------------------
/// Support C++-style iterators
//-----------------------------------------------------------------------------
    NodeConstChildIterator begin() const {
        return cbegin();
    }
    NodeConstChildIterator end() const {
        return cend();
    }

    NodeConstChildIterator cbegin() const;
    NodeConstChildIterator cend() const;

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

template<typename Reference, typename Pointer, typename Iter>
std::string NodeChildIteratorBase<Reference, Pointer, Iter>::name() const {
    return NodeConstIterator(m_parent, m_index + 1).name();
}
}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif

