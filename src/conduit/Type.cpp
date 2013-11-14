///
/// file: Type.cpp
///

#include "Type.h"

namespace conduit
{

// create storage for these guys
//uint32_dtype  = ValueType(BaseType::UINT32_T,1,0,0,sizeof(uint32);
//float64_dtype = ValueType(BaseType::FLOAT64_T,1,0,0,sizeof(float64));


///============================================
/// BaseType
///============================================

///============================================
BaseType::BaseType()
: m_id(EMPTY_T)
{}

///============================================
BaseType::BaseType(index_t id)
: m_id(id)
{}

///============================================
BaseType::~BaseType()
{}

///============================================     
index_t 
BaseType::type_name_to_id(const std::string &dtype_name)
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
BaseType::type_id_to_name(index_t dtype_id)
{
    if(dtype_id      == EMPTY_T)   return "[empty]";
    else if(dtype_id == NODE_T)    return "Node";
    else if(dtype_id == LIST_T)    return "List";
    else if(dtype_id == UINT32_T)  return "uint32";
    else if(dtype_id == UINT64_T)  return "uint64";
    else if(dtype_id == FLOAT64_T) return "float64";
    return "[empty]";
}

///============================================
///
/// ValueType
///
///============================================
///============================================
ValueType::ValueType()
: BaseType(BaseType::EMPTY_T),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0)
{}

///============================================
ValueType::ValueType(index_t dtype_id,
                     index_t num_elements,
                     index_t offset,
                     index_t stride,
                     index_t element_bytes)
: BaseType(dtype_id),
  m_num_ele(num_elements),
  m_offset(offset),
  m_stride(stride),
  m_ele_bytes(element_bytes)
{}
              
///============================================
ValueType::~ValueType()
{}
  
///============================================
index_t     
ValueType::element_index(index_t idx) const
{
    return m_offset + m_stride * idx;
}

///============================================
///
/// uint32_array
///
///============================================
///============================================
uint32_array::uint32_array(index_t num_elements,
                           index_t offset,
                           index_t stride,
                           index_t element_bytes=4)
    : ValueType(ValueType::UINT32_T,
                num_elements,
                offset,
                stride,
                element_bytes)
{}

uint32_array::~uint32_array()
{}


///============================================
///
/// float64_array
///
///============================================
///============================================
    float64_array::float64_array(index_t num_elements,
                                 index_t offset,
                                 index_t stride,
                                 index_t element_bytes=8)
        : ValueType(ValueType::FLOAT64_T,
                    num_elements,
                    offset,
                    stride,
                    element_bytes)
    {}

    float64_array::~float64_array()
    {}


///============================================
///
/// ListType
///
///============================================

///============================================
ListType::ListType()
: BaseType(BaseType::LIST_T)
{}

///============================================
ListType::ListType(Node *entry_schema,
                   index_t num_entries)
: BaseType(BaseType::LIST_T)
{
    m_schemas.push_back(entry_schema);
    m_num_entries.push_back(num_entries);
}
    
///============================================
ListType::ListType(const std::vector<Node *>   &entry_schemas,
                   const std::vector<index_t>  &num_entries)
: BaseType(BaseType::LIST_T)
{
    m_schemas = entry_schemas;
    m_num_entries = num_entries;
}


///============================================
ListType::~ListType()
{}

///============================================
/// Helpers 
///============================================
 
///============================================
BaseType Type(const std::string &dtype_name,
              index_t num_elements,
              index_t offset,
              index_t stride,
              index_t element_bytes)
{
    return Type(BaseType::type_name_to_id(dtype_name),
                num_elements, 
                offset, 
                stride, 
                element_bytes);
}

///============================================
BaseType Type(index_t dtype_id,
              index_t num_elements,
              index_t offset,
              index_t stride,
              index_t element_bytes)
{
    return ValueType(dtype_id,
                     num_elements, 
                     offset, 
                     stride, 
                     element_bytes);
}



///============================================
BaseType Type(Node *obj_schema,
              index_t num_entries)
{
    return ListType(obj_schema,num_entries);
}

///============================================
BaseType Type(std::vector<Node*>   obj_schemas,
              std::vector<index_t> num_entries)
{    
    return ListType(obj_schemas,num_entries);
}
              



};
