///
/// file: Type.cpp
///

#include "Type.h"

namespace conduit
{

// create storage for these guys
DataType DataType::uint32_dtype(DataType::UINT32_T,1,0,0,sizeof(uint32));
DataType DataType::float64_dtype(DataType::FLOAT64_T,1,0,0,sizeof(float64));


///============================================
/// DataType
///============================================

///============================================
DataType::DataType()
: m_id(DataType::EMPTY_T),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0)
{}

DataType::DataType(index_t dtype_id,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes)
: m_id(dtype_id),
  m_num_ele(num_elements),
  m_offset(offset),
  m_stride(stride),
  m_ele_bytes(element_bytes)
{}

DataType::DataType(const DataType& value)
: m_id(value.m_id),
  m_num_ele(value.m_num_ele),
  m_offset(value.m_offset),
  m_stride(value.m_stride),
  m_ele_bytes(value.m_ele_bytes)
{}
///============================================
DataType::DataType(index_t id)
: m_id(id),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0)
{}

///============================================
DataType::~DataType()
{}

index_t     
DataType::element_index(index_t idx) const
{
    return m_offset + m_stride * idx;
}

index_t
DataType::total_bytes() const
{
   return m_ele_bytes * m_num_ele;
}

///============================================     
index_t 
DataType::type_name_to_id(const std::string &dtype_name)
{
    if(dtype_name      == "[empty]") return EMPTY_T;
    else if(dtype_name == "Node")    return NODE_T;
    else if(dtype_name == "List")    return LIST_T;
    else if(dtype_name == "uint32")  return UINT32_T;
    else if(dtype_name == "uint64")  return UINT64_T;
    else if(dtype_name == "float64") return FLOAT64_T;
    return EMPTY_T;
}

///============================================
std::string 
DataType::type_id_to_name(index_t dtype_id)
{
    if(dtype_id      == EMPTY_T)   return "[empty]";
    else if(dtype_id == NODE_T)    return "Node";
    else if(dtype_id == LIST_T)    return "List";
    else if(dtype_id == UINT32_T)  return "uint32";
    else if(dtype_id == UINT64_T)  return "uint64";
    else if(dtype_id == FLOAT64_T) return "float64";
    return "[empty]";
}

index_t
DataType::size_of_type_id(index_t dtype)
{
    int size;
    switch (dtype) {
        case EMPTY_T : {
            size = 0;
            break;
        }
        case UINT32_T : {
            size = sizeof(uint32);
            break;
        }
        case UINT64_T : {
            size = sizeof(uint64);
            break;
        }
        case FLOAT64_T : {
            size = sizeof(float64);
            break;
        }
        default : {
            size = 0;
            break;
        }
    }
   return size;
}


///============================================
///
/// ListType
///
///============================================

// ///============================================
// ListType::ListType()
// : BaseType(BaseType::LIST_T)
// {}
// 
// ///============================================
// ListType::ListType(Node *entry_schema,
//                    index_t num_entries)
// : BaseType(BaseType::LIST_T)
// {
//     m_schemas.push_back(entry_schema);
//     m_num_entries.push_back(num_entries);
// }
//     
// ///============================================
// ListType::ListType(const std::vector<Node *>   &entry_schemas,
//                    const std::vector<index_t>  &num_entries)
// : BaseType(BaseType::LIST_T)
// {
//     m_schemas = entry_schemas;
//     m_num_entries = num_entries;
// }
// 
// 
// ///============================================
// ListType::~ListType()
// {}

///============================================
/// Helpers 
///============================================
 
///============================================
DataType Type(const std::string &dtype_name,
              index_t num_elements,
              index_t offset,
              index_t stride,
              index_t element_bytes)
{
    return Type(DataType::type_name_to_id(dtype_name),
                num_elements, 
                offset, 
                stride, 
                element_bytes);
}

///============================================
DataType Type(index_t dtype_id,
              index_t num_elements,
              index_t offset,
              index_t stride,
              index_t element_bytes)
{
    return DataType(dtype_id,
                     num_elements, 
                     offset, 
                     stride, 
                     element_bytes);
}


// 
// ///============================================
// BaseType Type(Node *obj_schema,
//               index_t num_entries)
// {
//     return ListType(obj_schema,num_entries);
// }
// 
// ///============================================
// BaseType Type(std::vector<Node*>   obj_schemas,
//               std::vector<index_t> num_entries)
// {    
//     return ListType(obj_schemas,num_entries);
// }
              



};
