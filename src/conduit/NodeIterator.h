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
    virtual  ~NodeIterator();
 
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
