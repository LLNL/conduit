///
/// file: DataArray.h
///

#include "conduit.h"

namespace conduit
{

class Node
{
public:
    Node(); // empty node
    Node(const Node& node); // copy 
    Node(const Schema& schema);
    
    Node &operator=(const Node&);
    Node &operator=(uint32);
    Node &operator=(uint64);
    Node &operator=(float64);
    Node &operator=(std::string &);
    Node &operator=(const char *);
    
    virtual  ~Node();

    bool              operator==(const Node &obj) const;

    Node             &operator[](const std::string &key);
    Node             &operator[](index_t idx);

    Node             *value(const std::string &key);
    const Node       *value(const std::string &key) const;

    Node             *value(index_t idx);
    const Node       *value(index_t idx) const;

    bool              has_key(const std::string &key) const;
    index_t           index(const std::string &key) const;
    std::string       key(index_t) const;
    void              keys(std::vector<std::string> &keys) const;

};

};