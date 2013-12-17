///
/// file: Node.h
///

#ifndef conduit_node_h__
#define conduit_node_h__

#include "Core.h"
#include "Endianness.h"
#include "Type.h"
#include "Array.h"

#include <map>
#include <vector>
#include <string>
#include <sstream>

#include "rapidjson/document.h"

namespace conduit
{

class Node
{
public:    
    
    /* Constructors */
    Node(); // empty node
    Node(const Node &node);
    explicit Node(const DataType &dtype, bool locked=false);
    Node(void *data, const std::string &schema);
    Node(void *data, const Node *schema);
    Node(void *data, const DataType &dtype);


    explicit Node(bool   data);
    explicit Node(int8   data);
    explicit Node(int16  data);
    explicit Node(int32  data);
    explicit Node(int64  data);

    explicit Node(uint8   data);
    explicit Node(uint16  data);
    explicit Node(uint32  data);
    explicit Node(uint64  data);

    explicit Node(float32 data);
    explicit Node(float64 data);

    explicit Node(const std::vector<int8>  &data);
    explicit Node(const std::vector<int16>  &data);    
    explicit Node(const std::vector<int32>  &data);
    explicit Node(const std::vector<int64>  &data);

    explicit Node(const std::vector<uint8>  &data);
    explicit Node(const std::vector<uint16>  &data);    
    explicit Node(const std::vector<uint32>  &data);
    explicit Node(const std::vector<uint64>  &data);
    
    explicit Node(const std::vector<float32>  &data);
    explicit Node(const std::vector<float64>  &data);
    
    explicit Node(const int8_array  &data);
    explicit Node(const int16_array &data);
    explicit Node(const int32_array &data);
    explicit Node(const int64_array &data);

    explicit Node(const uint8_array  &data);
    explicit Node(const uint16_array &data);
    explicit Node(const uint32_array &data);
    explicit Node(const uint64_array &data);

    explicit Node(const float32_array &data);
    explicit Node(const float64_array &data);
    
    explicit Node(const std::string  &data);

    virtual  ~Node();

    /// For each dtype:
    ///  constructor: explicit Node({DTYPE}  data);
    ///  assign op:  Node &operator=({DTYPE} data);
    ///  setter: void set({DTYPE} data);
    /// accessor: {DTYPE} as_{DTYPE};
    
    /* Setters */
    void set(const Node& data);
    void set(const DataType &data);

    void set(void* data, const std::string &schema);
    void set(void* data, const Node *schema);
    void set(void* data, const DataType &dtype);

    void set(bool data);
    
    void set(int8 data);
    void set(int16 data);
    void set(int32 data);
    void set(int64 data);

    void set(uint8 data);
    void set(uint16 data);
    void set(uint32 data);
    void set(uint64 data);

    void set(float32 data);
    void set(float64 data);

    void set(const std::vector<int8>   &data);
    void set(const std::vector<int16>  &data);
    void set(const std::vector<int32>  &data);
    void set(const std::vector<int64>  &data);

    void set(const std::vector<uint8>   &data);
    void set(const std::vector<uint16>  &data);
    void set(const std::vector<uint32>  &data);
    void set(const std::vector<uint64>  &data);

    void set(const std::vector<float32> &data);
    void set(const std::vector<float64> &data);

    void set(const int8_array  &data);
    void set(const int16_array &data);
    void set(const int32_array &data);
    void set(const int64_array &data);

    void set(const uint8_array  &data);
    void set(const uint16_array &data);
    void set(const uint32_array &data);
    void set(const uint64_array &data);

    void set(const float32_array &data);
    void set(const float64_array &data);
    
    void set(const std::string &data);
                
    /* Assignment ops */
    Node &operator=(const Node &node);
    Node &operator=(DataType dtype);

    Node &operator=(bool data);

    Node &operator=(int8 data);
    Node &operator=(int16 data);
    Node &operator=(int32 data);
    Node &operator=(int64 data);

    Node &operator=(uint8 data);
    Node &operator=(uint16 data);
    Node &operator=(uint32 data);
    Node &operator=(uint64 data);

    Node &operator=(float32 data);    
    Node &operator=(float64 data);

    Node &operator=(const std::vector<int8>   &data);
    Node &operator=(const std::vector<int16>   &data);
    Node &operator=(const std::vector<int32>   &data);
    Node &operator=(const std::vector<int64>   &data);

    Node &operator=(const std::vector<uint8>   &data);
    Node &operator=(const std::vector<uint16>   &data);
    Node &operator=(const std::vector<uint32>   &data);
    Node &operator=(const std::vector<uint64>   &data);

    Node &operator=(const std::vector<float32>  &data);
    Node &operator=(const std::vector<float64>  &data);

    Node &operator=(const int8_array  &data);
    Node &operator=(const int16_array &data);
    Node &operator=(const int32_array &data);
    Node &operator=(const int64_array &data);

    Node &operator=(const uint8_array  &data);
    Node &operator=(const uint16_array &data);
    Node &operator=(const uint32_array &data);
    Node &operator=(const uint64_array &data);

    Node &operator=(const float32_array &data);
    Node &operator=(const float64_array &data);

    Node &operator=(const std::string &data);

    /* Template Access (experimental) */
    template<typename T>
    void setpp(T data);
    
    template<typename T>
    T getpp(void) const;
    
    template<typename T>
    T getpp(void *data) const;

    /* Info */
    index_t total_bytes() const;
    const DataType    &dtype() const { return m_dtype;}

    /*schema access */
    std::string schema() const;
    void        schema(std::ostringstream &oss) const;
    void        lock_schema();
    void        unlock_schema();
    bool        is_schema_locked() const {return m_locked;}

    /* serialization */
    void        serialize(std::vector<uint8> &data, bool compact=true) const;
    void        serialize(uint8 *data, index_t curr_offset, bool compact=true) const;


    // TODO:
    // update() will add entries from n to current Node (like python dict update) 
    // void              update(const Node &n);  
    bool             compare(const Node &n, Node &cmp_results) const;
    bool             operator==(const Node &n) const;
    
    bool             is_empty() const;
    
    // these guys don't modify map structure, if a path doesn't exit
    // they will return an Empty Locked Node (we could also throw an exception)
    
    Node             &get(const std::string &path);
    Node             &get(index_t idx);

    const Node       &get(const std::string &path) const;
    const Node       &get(index_t idx) const;
    
    Node             &fetch(const std::string &path);
    Node             &fetch(index_t idx);

    template<typename TYPE>
    void             push_back(TYPE data);

    bool             has_path(const std::string &path) const;
    void             paths(std::vector<std::string> &paths,bool expand=false) const;


    Node             &operator[](const std::string &path);
    Node             &operator[](const index_t idx);
    const Node       &operator[](const std::string &path) const;
    const Node       &operator[](const index_t idx) const;


    int64            to_int()   const;
    uint64           to_uint()  const;
    float64          to_float() const;
    
    std::string      to_string() const;

    bool             as_bool()   const  { return *((bool*)element_pointer(0));}

    int8             as_int8()   const  { return *((int8*)element_pointer(0));}
    int16            as_int16()  const  { return *((int16*)element_pointer(0));}
    int32            as_int32()  const  { return *((int32*)element_pointer(0));}
    int64            as_int64()  const  { return *((int64*)element_pointer(0));}

    uint8            as_uint8()   const { return *((uint8*)element_pointer(0));}
    uint16           as_uint16()  const { return *((uint16*)element_pointer(0));}
    uint32           as_uint32()  const { return *((uint32*)element_pointer(0));}
    uint64           as_uint64()  const { return *((uint64*)element_pointer(0));}

    float32          as_float32() const { return *((float32*)element_pointer(0));}
    float64          as_float64() const { return *((float64*)element_pointer(0));}

    int8            *as_int8_ptr()     { return (int8*)element_pointer(0);}
    int16           *as_int16_ptr()    { return (int16*)element_pointer(0);}
    int32           *as_int32_ptr()    { return (int32*)element_pointer(0);}
    int64           *as_int64_ptr()    { return (int64*)element_pointer(0);}

    uint8           *as_uint8_ptr()    { return (uint8*)element_pointer(0);}
    uint16          *as_uint16_ptr()   { return (uint16*)element_pointer(0);}
    uint32          *as_uint32_ptr()   { return (uint32*)element_pointer(0);}
    uint64          *as_uint64_ptr()   { return (uint64*)element_pointer(0);}
        
    float32         *as_float32_ptr()  { return (float32*)element_pointer(0);}
    float64         *as_float64_ptr()  { return (float64*)element_pointer(0);}
    
    int8_array       as_int8_array()   { return int8_array(m_data,m_dtype);}
    int16_array      as_int16_array()  { return int16_array(m_data,m_dtype);}
    int32_array      as_int32_array()  { return int32_array(m_data,m_dtype);}
    int64_array      as_int64_array()  { return int64_array(m_data,m_dtype);}

    uint8_array      as_uint8_array()  { return uint8_array(m_data,m_dtype);}
    uint16_array     as_uint16_array() { return uint16_array(m_data,m_dtype);}
    uint32_array     as_uint32_array() { return uint32_array(m_data,m_dtype);}
    uint64_array     as_uint64_array() { return uint64_array(m_data,m_dtype);}

    float32_array    as_float32_array() { return float32_array(m_data,m_dtype);}
    float64_array    as_float64_array() { return float64_array(m_data,m_dtype);}

    
    char            *as_bytestr() {return (char *)element_pointer(0);}

    static Node     &empty() {return m_empty;}
    
private:
    // used to return something for the "static and or locked case"
    static Node        m_empty;


    void             init(const DataType &dtype);
    void             cleanup(); //dalloc
    
    void             set_lock(bool value);
    void             enforce_lock() const;
    
    void             walk_schema(void *data, 
                                 const std::string &schema);

    void             walk_schema(void *data, 
                                 const rapidjson::Value &jvalue, 
                                 index_t curr_offset);
    
    static void      split_path(const std::string &path,
                                std::string &curr,
                                std::string &next);

    void             enforce_lock();
    
    void            *element_pointer(index_t idx)
                     {return static_cast<char*>(m_data) + m_dtype.element_index(idx);};
    const void      *element_pointer(index_t idx) const 
                     {return static_cast<char*>(m_data) + m_dtype.element_index(idx);};


    void     *m_data;
    bool      m_alloced;
    DataType  m_dtype;
    // TODO: holds structure for true nodes + lists
    void     *m_obj_data;
    bool      m_locked;

    // for true nodes
    std::map<std::string, Node>         &entries();
    std::vector<Node>                   &list();

    const std::map<std::string, Node>   &entries() const;
    const std::vector<Node>             &list() const;


    std::vector<Node> m_list_data;
    std::map<std::string, Node> m_entries;
};

// TODO: Explicit temp inst in the c file. 

template<typename TYPE>
void Node::push_back(TYPE data)
{
   if (m_dtype.id() != DataType::LIST_T) {
       m_dtype.reset(DataType::LIST_T);
   }
   list().push_back(Node(data));
}

template<typename T>
void Node::setpp(T data)
{
   // TODO, if DataType::Traits shouldn't be private for this case
   // TODO check for compatible, don't always re-init
   init(DataType::default_dtype(DataType::Traits<T>::data_type));
   *((T*)m_data) = data;
}

template<typename T>
T Node::getpp(void) const
{
   // TODO some kind of checking here.. is m_data valid? right type?
   return *(T const *)( ((char const *)m_data) + m_dtype.element_index(0));
}

template<typename T>
T Node::getpp(void *data) const
{
   // TODO some kind of checking here.. is m_data valid? right type?
   return *(T const *)( ((char const *)data) + m_dtype.element_index(0));
}


}


#endif
