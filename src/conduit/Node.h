///
/// file: Node.h
///


#include "Core.h"
#include "Type.h"

namespace conduit
{

class Node
{
public:
    Node(); // empty node
    Node(const Node &node); // copy 
    Node(void *data, const Node *schema);
    Node(void *data, const BaseType *dtype);
    
    Node(BaseType dtype);
    Node(uint32  data);
    Node(float64 data);

    Node(uint32_array  *data);
    Node(float64_array *data);

    Node(const uint32_array  &data);
    Node(const float64_array &data);

    virtual  ~Node();
  
    void set(const &Node data);
    void set(BaseType data);

    void set(uint32 data);
    void set(float64 data);

    void set(const uint32_array  &data);
    void set(const float64_array &data);
    
    Node &operator=(const Node &node);

    Node &operator=(BaseType dtype);

    Node &operator=(uint32 data);
    Node &operator=(float64 data);

    Node &operator=(uint32_array  *data);
    Node &operator=(float64_array *data);

    Node &operator=(uint32_array  &data);
    Node &operator=(float64_array &data);

    std::string to_schema() const;

    const BaseType    &dtype() const { return *m_dtype;}
    bool              operator==(const Node &obj) const;
    // TODO: we will likly need const variants of these methods
                      
    Node             &fetch(const std::string &path);
    bool             has_path(const std::string &path) const;
    void             paths(std::vector<std::string> &paths,bool expand=false) const;
    
    Node             &operator[](const std::string &path)
                      {return fetch(path);}

    index            to_integer() const;    
    float64          to_real()    const;
        
    uint32           as_uint32()  const { return *((uint32*)element_pointer(0));}
    float64          as_float64() const { return *((float64*)element_pointer(0));}
    

    uint32_array     as_uint32_array()   { return uint32_array(element_pointer(0),m_dtype);}
    float64_array    as_float64_array()  { return float64_array(element_pointer(0),m_dtype);}

    
private:
    void             init(const BaseType &dtype);
    void             init(BaseType *dtype);
    void             cleanup(); //dalloc 
    

    index            element_index(index idx) const;
    void            *element_pointer(index idx){return m_data[element_index(idx)]};
    const void      *element_pointer(index idx) const {return m_data[element_index(idx)]};
    std::map<Node*>  entries();

    bool      m_alloced_data;
    bool      m_alloced_dtype;
    void     *m_data;
    BaseType *m_dtype;
    
};

};