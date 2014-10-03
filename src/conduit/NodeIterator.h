/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: NodeIterator.h
///

#ifndef __CONDUIT_NODE_ITERATOR_H
#define __CONDUIT_NODE_ITERATOR_H

#include "Error.h"
#include "Node.h"
#include "Utils.h"

namespace conduit
{
    

class NodeIterator
{
public:


    /* Constructors */
    NodeIterator(); // empty itr
    NodeIterator(Node *node,index_t idx=0);
    NodeIterator(const NodeIterator &itr);
    ~NodeIterator();
 
    /* Assignment ops */
    NodeIterator &operator=(const NodeIterator &itr);
 
    /* Iter Values */
    std::string path()  const;
    index_t     index() const;
    Node       &node();
    void        to_front();

    /* Iter Fwd Control */
    bool        has_next() const;
    Node       &next();
    Node       &peak_next();

    /* Iter Rev Control */
    bool        has_previous() const;
    Node       &previous();
    Node       &peak_previous();
    void        to_back();
    
    void        info(Node &res) const;
    
private:
    Node    *m_node;
    index_t  m_index;
    index_t  m_num_ele; // for easier index calcs in the iterator
    
};

}


#endif
