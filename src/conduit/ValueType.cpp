///
/// file: ValueType.cpp
///

#include "ValueType.h"

namespace conduit
{

// 
// 
// ValueType::ValueType(index_t dtype)
// : m_id(dtype)
// {}
//     
// ValueType::ValueType(const std::string &dtype_name)
// {
//     m_id = ValueType::name_to_id(dtype_name);
// }
//     
// ValueType::ValueType(const ValueType   &dtype)
// : m_id(dtype.m_id)
// {}
// 
// // ValueType    ValueType::&operator=(ValueType &dtype)
// // {}
// // ValueType    ValueType::&operator=(const ValueType &dtype)
// // {}
    
index_t     ValueType::id(const std::string &name)
{
    if(name == "[empty]")      return EMPTY_T;
    else if(name == "Node")    return NODE_T;
    else if(name == "List")    return LIST_T;
    else if(name == "uint32")  return UINT32_T;
    else if(name == "uint64")  return UINT64_T;
    else if(name == "float64") return FLOAT64_T;
    return UNKNOWN_T;
}

std::string ValueType::name(index_t dtype)
{
    if(dtype == EMPTY_T)        return "[empty]";
    else if(dtype == NODE_T)    return "Node";
    else if(dtype == LIST_T)    return "List";
    else if(dtype == UINT32_T)  return "uint32";
    else if(dtype == UINT64_T)  return "uint64";
    else if(dtype == FLOAT64_T) return "float64";
    return "[unknown]";
}

};
