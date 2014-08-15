///
/// file: Node.h
///

#ifndef __CONDUIT_NODE_H
#define __CONDUIT_NODE_H

#include "Core.h"
#include "Error.h"
#include "Endianness.h"
#include "DataType.h"
#include "DataArray.h"
#include "Schema.h"
#include "Generator.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>


namespace conduit
{
    
class Generator;

class Node
{
public:


    /* Constructors */
    Node(); // empty node
    Node(const Node &node);
    explicit Node(const DataType &dtype);

    Node(const Generator &gen);
    
    // convience interface:
    Node(const std::string &json_schema, void *data);

    Node(const Schema &schema);
    Node(const Schema &schema, void *data);
    Node(const Schema &schema, const std::string &stream_path, bool mmap=false);    
    //Node(Schema &schema, std::ifstream &ifs);
    Node(const DataType &dtype, void *data);
        
    explicit Node(bool8  data);

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

    explicit Node(const std::vector<bool8>  &data);

    explicit Node(const std::vector<int8>   &data);
    explicit Node(const std::vector<int16>  &data);    
    explicit Node(const std::vector<int32>  &data);
    explicit Node(const std::vector<int64>  &data);

    explicit Node(const std::vector<uint8>   &data);
    explicit Node(const std::vector<uint16>  &data);    
    explicit Node(const std::vector<uint32>  &data);
    explicit Node(const std::vector<uint64>  &data);
    
    explicit Node(const std::vector<float32>  &data);
    explicit Node(const std::vector<float64>  &data);

    explicit Node(const bool8_array  &data);    

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

    void reset();
    void load(const Schema &schema, const std::string &stream_path);
    void mmap(const Schema &schema, const std::string &stream_path);

    
    /// For each dtype:
    ///  constructor: explicit Node({DTYPE}  data);
    ///  assign op:  Node &operator=({DTYPE} data);
    ///  setter: void set({DTYPE} data);
    /// accessor: {DTYPE} as_{DTYPE};
      
    /* Setters */
    void set(const Node& data);
    void set(const Node& node, Schema* schema);
    void set(const DataType& dtype);

    void set(const Schema &schema);
    void set(const Schema &schema, void* data);
    void set(const DataType &dtype, void* data);

    void set(bool8 data);
    
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

    void set(const std::vector<bool8>   &data);
    
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

    void set(const bool8_array  &data);

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

    // bytestr use cases:
    void set(const char* data);    
    void set(const std::string &data);
                
    /* Assignment ops */
    Node &operator=(const Node &node);
    Node &operator=(DataType dtype);

    Node &operator=(bool8 data);

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

    // bytestr use cases:
    Node &operator=(const char* data);
    Node &operator=(const std::string &data);

    /*schema access */
    const Schema     &schema() const { return *m_schema;}   

    Schema           *schema_pointer() {return m_schema;}   

    /* parent access */
    bool             has_parent() const {return m_parent != NULL;}
    void             set_parent(Node *parent) { m_parent = parent;}
    Node            *parent() {return m_parent;}
    
    /* Info */
    index_t           total_bytes() const { return m_schema->total_bytes();}
    const DataType   &dtype() const       { return m_schema->dtype();}
    
    /* serialization */
    void        serialize(std::vector<uint8> &data, bool compact=true) const;
    void        serialize(uint8 *data, index_t curr_offset, bool compact=true) const;

    void        serialize(const std::string &stream_path, bool compact=true) const;
    // In the future, support our own IOStreams (which will provide single interface 
    // for bin,hdf,silo end-points.
    void        serialize(std::ofstream &ofs, bool compact=true) const;

    /// TODO:
    /// update() will add entries from n to current Node (like python dict update) 
    /// void        update(const Node &n);  
    //  bool        compare(const Node &n, Node &cmp_results) const;
    //  bool        operator==(const Node &n) const;
    
    ///
    /// Entry Access
    ///    
    // the `fetch' methods do modify map structure if a path doesn't exists
    Node             &fetch(const std::string &path);
    Node             &fetch(index_t idx);
    
    Node             *fetch_pointer(const std::string &path);
    Node             *fetch_pointer(index_t idx);

    void append(Node *node)
        {m_children.push_back(node);}

    void append()
        {list_append(Node());}

    void append(const Node &node)
        {list_append(Node(node));}

    void append(const DataType &data)
        {list_append(Node(data));}

    void append(bool8 data)
        {list_append(Node(data));}        
    void append(int8 data)
        {list_append(Node(data));}
    void append(int16 data)
        {list_append(Node(data));}
    void append(int32 data)
        {list_append(Node(data));}
    void append(int64 data)
        {list_append(Node(data));}

    void append(uint8 data)
        {list_append(Node(data));}
    void append(uint16 data)
        {list_append(Node(data));}
    void append(uint32 data)
        {list_append(Node(data));}
    void append(uint64 data)
        {list_append(Node(data));}
    void append(float32 data)
        {list_append(Node(data));}
    void append(float64 data)
        {list_append(Node(data));}

    void append(const std::vector<bool8>   &data)
        {list_append(Node(data));}

    void append(const std::vector<int8>   &data)
        {list_append(Node(data));}
    void append(const std::vector<int16>  &data)
        {list_append(Node(data));}
    void append(const std::vector<int32>  &data)
        {list_append(Node(data));}
    void append(const std::vector<int64>  &data)
        {list_append(Node(data));}

    void append(const std::vector<uint8>   &data)
        {list_append(Node(data));}
    void append(const std::vector<uint16>  &data)
        {list_append(Node(data));}
    void append(const std::vector<uint32>  &data)
        {list_append(Node(data));}
    void append(const std::vector<uint64>  &data)
        {list_append(Node(data));}
    void append(const std::vector<float32> &data)
        {list_append(Node(data));}
    void append(const std::vector<float64> &data)
        {list_append(Node(data));}

    void append(const bool8_array  &data)
        {list_append(Node(data));}

    void append(const int8_array  &data)
        {list_append(Node(data));}
    void append(const int16_array &data)
        {list_append(Node(data));}
    void append(const int32_array &data)
        {list_append(Node(data));}
    void append(const int64_array &data)
        {list_append(Node(data));}

    void append(const uint8_array  &data)
        {list_append(Node(data));}
    void append(const uint16_array &data)
        {list_append(Node(data));}
    void append(const uint32_array &data)
        {list_append(Node(data));}
    void append(const uint64_array &data)
        {list_append(Node(data));}

    void append(const float32_array &data)
        {list_append(Node(data));}
    void append(const float64_array &data)
        {list_append(Node(data));}
    
    void append(const std::string &data)
        {list_append(Node(data));}

    index_t number_of_entries() const;
    void    remove(index_t idx);
    void    remove(const std::string &path);
    
    bool    has_path(const std::string &path) const;
    void    paths(std::vector<std::string> &paths,bool walk=false) const;


    // these support the map and list interfaces
    Node             &operator[](const std::string &path);
    Node             &operator[](const index_t idx);

    // TODO crs methods to all types
    int64            to_int64()   const;
    uint64           to_uint64()  const;
    float64          to_float64() const;
    index_t          to_index_t() const;
        
    std::string      to_json(bool simple=false,
                             index_t indent=0) const;
    void             to_json(std::ostringstream &oss,
                             bool simple=false,
                             index_t indent=0) const;

    bool8            as_bool8()   const { return *((bool8*)element_pointer(0));}

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

    bool8           *as_bool8_ptr()    { return (bool8*)element_pointer(0);}

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
    
    bool8_array      as_bool8_array()  { return bool8_array(m_data,dtype());}
    
    int8_array       as_int8_array()   { return int8_array(m_data,dtype());}
    int16_array      as_int16_array()  { return int16_array(m_data,dtype());}
    int32_array      as_int32_array()  { return int32_array(m_data,dtype());}
    int64_array      as_int64_array()  { return int64_array(m_data,dtype());}

    uint8_array      as_uint8_array()  { return uint8_array(m_data,dtype());}
    uint16_array     as_uint16_array() { return uint16_array(m_data,dtype());}
    uint32_array     as_uint32_array() { return uint32_array(m_data,dtype());}
    uint64_array     as_uint64_array() { return uint64_array(m_data,dtype());}

    float32_array    as_float32_array() { return float32_array(m_data,dtype());}
    float64_array    as_float64_array() { return float64_array(m_data,dtype());}

    bool8_array      as_bool8_array() const { return bool8_array(m_data,dtype());}
    
    int8_array       as_int8_array()  const { return int8_array(m_data,dtype());}
    int16_array      as_int16_array() const { return int16_array(m_data,dtype());}
    int32_array      as_int32_array() const { return int32_array(m_data,dtype());}
    int64_array      as_int64_array() const { return int64_array(m_data,dtype());}

    uint8_array      as_uint8_array()  const { return uint8_array(m_data,dtype());}
    uint16_array     as_uint16_array() const { return uint16_array(m_data,dtype());}
    uint32_array     as_uint32_array() const { return uint32_array(m_data,dtype());}
    uint64_array     as_uint64_array() const { return uint64_array(m_data,dtype());}

    float32_array    as_float32_array() const { return float32_array(m_data,dtype());}
    float64_array    as_float64_array() const { return float64_array(m_data,dtype());}

    char            *as_bytestr() {return (char *)element_pointer(0);}
    const char      *as_bytestr() const {return (const char *)element_pointer(0);}
    
    std::string      as_string() const {return std::string(as_bytestr());}

    // these were private
    void             set(Schema *schema_ptr);
                     Node(Schema *schema_ptr);
    void             set(Schema *schema_ptr, void *data_ptr);
    
                     Node(const Node &node,Schema *schema_ptr);
    
    
private:
    void             init(const DataType &dtype);

    void             allocate(index_t dsize); 
    void             allocate(const DataType &dtype); 
    void             mmap(const std::string &stream_path,index_t dsize);
    void             cleanup();
    void             release();
    
    void             walk_schema(const Schema &schema);

    void             walk_schema(const Schema &schema,
                                 void *data);

    void             walk_schema(Node   *node,
                                 Schema *schema,
                                 void   *data);
   
    void            *element_pointer(index_t idx)
                     {return static_cast<char*>(m_data) + dtype().element_index(idx);};
    const void      *element_pointer(index_t idx) const 
                     {return static_cast<char*>(m_data) + dtype().element_index(idx);};

    /* helper */
    void              init_defaults();
    void              init_list();
    void              init_object();
    void              list_append(const Node &node);

    Node                *m_parent;
    Schema              *m_schema;
    std::vector<Node*>   m_children;    

    // TODO DataContainer
    void     *m_data;
    bool      m_alloced;
    bool      m_mmaped;
    int       m_mmap_fd;
    index_t   m_mmap_size;

    // TODO: holds structure for objs + lists

    // for true nodes
//     std::map<std::string, Node>         &entries();
//     std::vector<Node>                   &list();
// 
//     const std::map<std::string, Node>   &entries() const;
//     const std::vector<Node>             &list() const;

    // TODO: These are currently alloced per node, even if we have a simple node type
    // Use m_obj_data w/ allocs in the future to reduce overhead. 
    // The entries() & list() helper funcs already provide a single point of access
    // so  change the storage shouldn't be very hard. 
//     std::vector<Node>           m_list_data;
//     std::map<std::string, Node> m_entries;
};

}


#endif
