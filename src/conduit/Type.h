///
/// file: Type.h
///

#ifndef __CONDUIT_TYPE_H
#define __CONDUIT_TYPE_H

#include "Core.h"
#include <vector>
#include <sstream>

namespace conduit
{

class Node;
///============================================
/// DataType
///============================================
class DataType
{
public:

   static DataType uint32_dtype;
   static DataType float64_dtype;

    typedef enum
    {
        EMPTY_T = 0, // default
        NODE_T,      // node
        LIST_T,      // list
        UINT32_T,    // uint32 and uint32_array
        UINT64_T,    // uint64 and uint64_array
        FLOAT64_T,   // float64 and float64_array
        BYTESTR_T,   // bytestr (incore c-string)
    } TypeEnum;
            
             DataType();
             DataType(index_t id);
             DataType(const DataType& type);
             DataType(const std::string &dtype_name,
                      index_t num_elements,
                      index_t offset,
                      index_t stride,
                      index_t element_bytes);
             DataType(index_t dtype_id,
                      index_t num_elements,
                      index_t offset,
                      index_t stride,
                      index_t element_bytes);
    virtual ~DataType();
    
    static index_t      type_name_to_id(const std::string &name);
    static std::string  type_id_to_name(index_t dtype);
    static index_t      size_of_type_id(index_t dtype);
    
    std::string         schema() const;
    void                schema(std::ostringstream &oss) const;
    
    index_t             id()  const { return m_id;}    
    index_t     total_bytes() const;

    index_t    number_of_elements()  const { return m_num_ele;}
    index_t    offset()              const { return m_offset;}
    index_t    stride()              const { return m_stride;}
    index_t    element_bytes()       const { return m_ele_bytes;}

    index_t    element_index(index_t idx) const;

private:
    index_t  m_id;
    index_t   m_num_ele;    // number of entries
    index_t   m_offset;     // bytes to start of array
    index_t   m_stride;     // bytes between start of current and start of next
    index_t   m_ele_bytes;  // bytes per element
    // TODO: future  index_t  m_endianness;

};

///============================================
/// ListType
///============================================
// this is a strange one ...// 
// class ListType: public BaseType
// {
// public:
//              ListType(); // fully dynamic
//              ListType(Node *entry_schema,
//                       index_t num_entries); 
//              ListType(const std::vector<Node *>  &entry_schemas,
//                       const std::vector<index_t> &num_entries); 
//     virtual ~ListType();
// 
// private:
//     std::vector<Node *>   m_schemas;
//     std::vector<index_t>  m_num_entries;
// 
// };


// these handle lists

// BaseType Type(const std::string &dtype_name,
//               const std::string &entry_schema, // Registry(entry_schema)
//               const std::string &num_ele_ref); // reference(len_ref)
// 
// BaseType Type(const std::string &dtype_name,
//               std::vector<Node*> obj_schemas,
//               std::vector<index> num_entries);
// 
// BaseType Type(const std::string &dtype_name,
//               Node *obj_schema,
//               index_tnum_entries);
// 
// BaseType Type(index_tdtype_id,
//               std::vector<Node*> obj_schemas,
//               std::vector<index> num_entries);
// 
// BaseType Type(index_tdtype_id,
//               Node *obj_schema,
//               index_tnum_entries);

// BaseType Type(Node    *obj_schema,
//               index_t  num_entries);
// 
// BaseType Type(std::vector<Node*>   obj_schemas,
//               std::vector<index_t> num_entries);
//               

// BaseType node_type();
// BaseType list_type(); // implies dynamic construction
// BaseType list_type(Node *entry_schema, index_tnum_entries); 
// BaseType list_type(List *entry_schemas); 
// 
// BaseType uint32_type(index_tnum_elements=1,
//                      index_toffset=0,
//                      index_tstride=sizeof(uint32),
//                      index_telement_bytes=sizeof(uint32));
// 
// BaseType uint64_type(index_tnum_elements=1,
//                      index_toffset=0,
//                      index_tstride=sizeof(uint64),
//                      index_telement_bytes=sizeof(uint64));
// 
// 
// BaseType float64_type(index_tnum_elements=1,
//                       index_toffset=0,
//                      index_tstride=sizeof(float64),
//                      index_telement_bytes=sizeof(float64));


};


#endif
