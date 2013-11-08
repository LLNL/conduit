///
/// file: DataArray.h
///

#include "conduit.h"
#include "ValueArray.h"

namespace conduit
{

class Node : public ValueArray
{
public:
    Node(); // empty node
    Node(const Node &node); // copy 
    
    Node(void *data, const Node *schema, bool expand);
    
    Node(ValueType dtype);
    
    Node(uint32  data);
    Node(uint64  data);
    Node(float64 data);

    Node(bytestr       *data,bool copy=false);
    Node(uint32_array  *data,bool copy=false);
    Node(uint64_array  *data,bool copy=false);
    Node(float64_array *data,bool copy=false);
    
    Node &operator=(const Node&);

    Node &operator=(ValueType dtype); // creates empty of this type (used for creating schemas)

    Node &operator=(uint32 data);
    Node &operator=(uint64 data);
    Node &operator=(float64 data);


    Node &operator=(bytestr       *data);
    Node &operator=(uint32_array  *data);
    Node &operator=(uint64_array  *data);
    Node &operator=(float64_array *data);

    Node &operator=(bytestr       &data);
    Node &operator=(uint32_array  &data);
    Node &operator=(uint64_array  &data);
    Node &operator=(float64_array &data);

        
    
    virtual  ~Node();

    bool              operator==(const Node &obj) const;

    // TODO: we will likley need const variants of these methods
    Node             &operator[](const std::string &path)
                      {return fetch(path);}
                      
    Node             &fetch(const std::string &path);
    
    uint32           fetch_uint32(const std::string &path);
    uint64           fetch_uint64(const std::string &path);
    float64          fetch_float64(const std::string &path);
    
    bytestr         *fetch_bytestr(std::string &path);
    uint32_array    *fetch_uint32_array(std::string &path);
    uint64_array    *fetch_uint64_array(std::string &path);
    float64_array   *fetch_float64_array(std::string &path);

    bool             has_path(const std::string &path) const;
    void             paths(std::vector<std::string> &paths,bool expand=false) const;
    
private:    
    // for pass though mode (1 schema n objects)
    Node                       *m_schema;

    std::map<std::string,Node>  m_entries;
    // the rest lives in ValueArray ...
    

};

};