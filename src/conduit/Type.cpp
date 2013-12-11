///
/// file: Type.cpp
///

#include "Type.h"

namespace conduit
{

// create storage for these common types:
DataType DataType::empty_dtype(DataType::EMPTY_T,
                               0,0,0,0,
                               Endianness::DEFAULT_T);

DataType DataType::uint32_dtype(DataType::UINT32_T,
                               1,0,
                               sizeof(uint32),
                               sizeof(uint32),
                               Endianness::DEFAULT_T);

DataType DataType::float64_dtype(DataType::FLOAT64_T,
                                 1,0,
                                 sizeof(float64),
                                 sizeof(float64),
                                 Endianness::DEFAULT_T);

DataType DataType::node_dtype(DataType::NODE_T);
DataType DataType::list_dtype(DataType::LIST_T);


///============================================
/// DataType
///============================================

///============================================
DataType::DataType()
: m_id(DataType::EMPTY_T),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0),
  m_endianness(Endianness::DEFAULT_T)
{}

///============================================ 
DataType::DataType(const std::string &dtype_name,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
: m_id(name_to_id(dtype_name)),
  m_num_ele(num_elements),
  m_offset(offset),
  m_stride(stride),
  m_ele_bytes(element_bytes),
  m_endianness(endianness)
{
}

///============================================ 
DataType::DataType(index_t dtype_id,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
                
: m_id(dtype_id),
  m_num_ele(num_elements),
  m_offset(offset),
  m_stride(stride),
  m_ele_bytes(element_bytes),
  m_endianness(endianness)
{}

///============================================ 
DataType::DataType(const DataType& value)
: m_id(value.m_id),
  m_num_ele(value.m_num_ele),
  m_offset(value.m_offset),
  m_stride(value.m_stride),
  m_ele_bytes(value.m_ele_bytes),
  m_endianness(value.m_endianness)
{}


///============================================
DataType::DataType(index_t id)
: m_id(id),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0),
  m_endianness(Endianness::DEFAULT_T)
{}

///============================================
DataType::~DataType()
{}

///============================================ 
void
DataType::reset(const DataType& dtype)
{
    m_id = dtype.m_id;
    m_num_ele = dtype.m_num_ele;
    m_offset = dtype.m_offset;
    m_stride = dtype.m_stride;
    m_ele_bytes = dtype.m_ele_bytes;
    m_endianness = dtype.m_endianness;
}

///============================================ 
void
DataType::reset(index_t dtype_id)
{
    m_id = dtype_id;
    m_num_ele = 0;
    m_offset = 0;
    m_stride = 0;
    m_ele_bytes = 0;
    m_endianness = Endianness::DEFAULT_T;
}

///============================================ 
void
DataType::reset(index_t dtype_id,
                index_t num_elements,
                index_t offset,
                index_t stride,
                index_t element_bytes,
                index_t endianness)
{
    m_id = dtype_id;
    m_num_ele = num_elements;
    m_offset = offset;
    m_stride = stride;
    m_ele_bytes = element_bytes;
    m_endianness = endianness;
}

///============================================ 
index_t     
DataType::element_index(index_t idx) const
{
    return m_offset + m_stride * idx;
}

///============================================ 
index_t
DataType::total_bytes() const
{
   // non compact case
   return m_stride * m_num_ele;
}

///============================================ 
index_t
DataType::total_bytes_compact() const
{
   return m_ele_bytes * m_num_ele;
}
    
///============================================
bool
DataType::compatible_storage(const DataType& dtype) const
{
    return ( (m_id == dtype.m_id ) &&
             (m_ele_bytes == dtype.m_ele_bytes) &&
             (total_bytes() == dtype.total_bytes()));
}

///============================================     
index_t 
DataType::name_to_id(const std::string &dtype_name)
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
DataType::id_to_name(index_t dtype_id)
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
DataType const &
DataType::default_dtype(index_t dtype_id)
{
   switch (dtype_id)
   {
        case UINT32_T : 
        {
            return DataType::uint32_dtype;
        }
        case FLOAT64_T : 
        {
            return DataType::float64_dtype;
        }
        default : 
        {
            return DataType::empty_dtype;
        }
    }
}


///============================================ 
DataType const &
DataType::default_dtype(const std::string &name)
{
    return default_dtype(name_to_id(name));
}


///============================================ 
std::string 
DataType::schema() const
{
    std::ostringstream oss;
    schema(oss);
    return oss.str();
}

///============================================ 
void
DataType::schema(std::ostringstream &oss) const
{
    oss << "{\"dtype\":";
    switch (m_id)
    {
        case EMPTY_T : 
        {
            oss << "\"[empty]\"";
            break;
        }
        case UINT32_T : 
        case FLOAT64_T : 
        {
            oss << "\"" << id_to_name(m_id) << "\"";
            oss << ", \"length\" : " << m_num_ele;
            oss << ", \"offset\" : " << m_offset;
            oss << ", \"stride\" : " << m_stride;
            oss << ", \"element_bytes\" : " << m_ele_bytes;

            std::string endian_str;

            if(m_endianness == Endianness::DEFAULT_T)
            {
                // find this machine's actual endianness
                endian_str = Endianness::id_to_name(Endianness::machine_default());
            }
            else
            {
                endian_str = Endianness::id_to_name(m_endianness);
            }
            oss << ", \"endianness\" : \"" << endian_str << "\"";            
            break;
        }
        default : 
        {
            oss << "\"[unknown]\"";
            break;
        }
    }
    oss << "}";
}

};
