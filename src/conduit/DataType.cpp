///
/// file: DataType.cpp
///

#include "DataType.h"
#include "Schema.h"

namespace conduit
{

///============================================
/// DataType::Objects
///============================================

// create storage for these common types:
DataType DataType::Objects::m_empty(DataType::EMPTY_T);
DataType DataType::Objects::m_object(DataType::OBJECT_T);
DataType DataType::Objects::m_list(DataType::LIST_T);

///============================================
/// DataType::Scalars
///============================================


DataType DataType::Scalars::m_bool8(DataType::BOOL8_T,
                                    1,0,
                                    sizeof(conduit::bool8),
                                    sizeof(conduit::bool8),
                                    Endianness::DEFAULT_T);

/* int default dtypes */
DataType DataType::Scalars::m_int8(DataType::INT8_T,
                                   1,0,
                                   sizeof(conduit::int8),
                                   sizeof(conduit::int8),
                                   Endianness::DEFAULT_T);

DataType DataType::Scalars::m_int16(DataType::INT16_T,
                                    1,0,
                                    sizeof(conduit::int16),
                                    sizeof(conduit::int16),
                                    Endianness::DEFAULT_T);

DataType DataType::Scalars::m_int32(DataType::INT32_T,
                                    1,0,
                                    sizeof(conduit::int32),
                                    sizeof(conduit::int32),
                                    Endianness::DEFAULT_T);

DataType DataType::Scalars::m_int64(DataType::INT64_T,
                                    1,0,
                                    sizeof(conduit::int64),
                                    sizeof(conduit::int64),
                                    Endianness::DEFAULT_T);

/* uint default dtypes */
DataType DataType::Scalars::m_uint8(DataType::UINT8_T,
                                    1,0,
                                    sizeof(conduit::uint8),
                                    sizeof(conduit::uint8),
                                    Endianness::DEFAULT_T);

DataType DataType::Scalars::m_uint16(DataType::UINT16_T,
                                     1,0,
                                     sizeof(conduit::uint16),
                                     sizeof(conduit::uint16),
                                     Endianness::DEFAULT_T);

DataType DataType::Scalars::m_uint32(DataType::UINT32_T,
                                     1,0,
                                     sizeof(conduit::uint32),
                                     sizeof(conduit::uint32),
                                     Endianness::DEFAULT_T);

DataType DataType::Scalars::m_uint64(DataType::UINT64_T,
                                     1,0,
                                     sizeof(conduit::uint64),
                                     sizeof(conduit::uint64),
                                     Endianness::DEFAULT_T);
/* float default dtypes */
DataType DataType::Scalars::m_float32(DataType::FLOAT32_T,
                                      1,0,
                                      sizeof(conduit::float32),
                                      sizeof(conduit::float32),
                                      Endianness::DEFAULT_T);

DataType DataType::Scalars::m_float64(DataType::FLOAT64_T,
                                     1,0,
                                     sizeof(conduit::float64),
                                     sizeof(conduit::float64),
                                     Endianness::DEFAULT_T);

///============================================
/// DataType::Arrays
///============================================

/* int array dtype helpers */
///============================================
DataType
DataType::Arrays::int8(index_t num_elements,
                       index_t offset,
                       index_t stride,
                       index_t element_bytes,
                       index_t endianness)
{
    return DataType(INT8_T,num_elements,offset,stride,element_bytes,endianness);
}

///============================================
DataType
DataType::Arrays::int16(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(INT16_T,num_elements,offset,stride,element_bytes,endianness);
}

///============================================
DataType 
DataType::Arrays::int32(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(INT32_T,num_elements,offset,stride,element_bytes,endianness);
}

///============================================
DataType
DataType::Arrays::int64(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(INT64_T,num_elements,offset,stride,element_bytes,endianness);
}

/* uint array dtype helpers */
///============================================
DataType
DataType::Arrays::uint8(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(UINT8_T,num_elements,offset,stride,element_bytes,endianness);
}
 
 ///============================================       
DataType
DataType::Arrays::uint16(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(UINT16_T,num_elements,offset,stride,element_bytes,endianness);
}

///============================================
DataType 
DataType::Arrays::uint32(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(UINT32_T,num_elements,offset,stride,element_bytes,endianness);
}

///============================================
DataType
DataType::Arrays::uint64(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(UINT64_T,num_elements,offset,stride,element_bytes,endianness);
}

/* float array dtype helpers */
///============================================
DataType 
DataType::Arrays::float32(index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    return DataType(FLOAT32_T,num_elements,offset,stride,element_bytes,endianness);
}

///============================================
DataType
DataType::Arrays::float64(index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    return DataType(FLOAT64_T,num_elements,offset,stride,element_bytes,endianness);
}



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
{}

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
//     std::cout << "offset = " << m_offset 
//               << " stride = " << m_stride << std::endl;
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
DataType::is_compatible(const DataType& dtype) const
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
    else if(dtype_name == "Object")  return OBJECT_T;
    else if(dtype_name == "List")    return LIST_T;
    else if(dtype_name == "bool8")   return BOOL8_T;
    else if(dtype_name == "int8")    return INT8_T;
    else if(dtype_name == "int16")   return INT16_T;
    else if(dtype_name == "int32")   return INT32_T;
    else if(dtype_name == "int64")   return INT64_T;
    else if(dtype_name == "uint8")   return UINT8_T;
    else if(dtype_name == "uint16")  return UINT16_T;
    else if(dtype_name == "uint32")  return UINT32_T;
    else if(dtype_name == "uint64")  return UINT64_T;
    else if(dtype_name == "float32") return FLOAT32_T;
    else if(dtype_name == "float64") return FLOAT64_T;
    else if(dtype_name == "bytestr") return BYTESTR_T;
    return EMPTY_T;
}

///============================================
std::string 
DataType::id_to_name(index_t dtype_id)
{
    if(dtype_id      == EMPTY_T)   return "[empty]";
    else if(dtype_id == OBJECT_T)  return "Object";
    else if(dtype_id == LIST_T)    return "List";
    else if(dtype_id == BOOL8_T)   return "bool8";
    /* ints */
    else if(dtype_id == INT8_T)    return "int8";
    else if(dtype_id == INT16_T)   return "int16";
    else if(dtype_id == INT32_T)   return "int32";
    else if(dtype_id == INT64_T)   return "int64";
    /* uints */
    else if(dtype_id == UINT8_T)   return "uint8";
    else if(dtype_id == UINT16_T)  return "uint16";
    else if(dtype_id == UINT32_T)  return "uint32";
    else if(dtype_id == UINT64_T)  return "uint64";
    /* floats */
    else if(dtype_id == FLOAT32_T) return "float32";
    else if(dtype_id == FLOAT64_T) return "float64";
    /* strs */
    else if(dtype_id == BYTESTR_T) return "bytestr";
    return "[empty]";
}


///============================================ 
DataType const &
DataType::default_dtype(index_t dtype_id)
{
   switch (dtype_id)
   {
        case OBJECT_T: return DataType::Objects::object();
        case LIST_T :  return DataType::Objects::list();
        case BOOL8_T : return DataType::Scalars::bool8();
        /* int types */
        case INT8_T :  return DataType::Scalars::int8();
        case INT16_T : return DataType::Scalars::int16();
        case INT32_T : return DataType::Scalars::int32();
        case INT64_T : return DataType::Scalars::int64();
        /* uint types */
        case UINT8_T :  return DataType::Scalars::uint8();
        case UINT16_T : return DataType::Scalars::uint16();
        case UINT32_T : return DataType::Scalars::uint32();
        case UINT64_T : return DataType::Scalars::uint64();
        /* float types */
        case FLOAT32_T : return DataType::Scalars::float32();
        case FLOAT64_T : return DataType::Scalars::float64();
        /* no default type for bytestr */
        default : 
        {
            return DataType::Objects::empty();
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
Schema              
DataType::schema() const
{
    return Schema(json_schema());
}
    
///============================================ 
std::string 
DataType::json_schema() const
{
    std::ostringstream oss;
    json_schema(oss);
    return oss.str();
}

///============================================ 
void
DataType::json_schema(std::ostringstream &oss, const std::string &value) const
{
    oss << "{\"dtype\":";
    if(m_id == EMPTY_T)
    {
        oss << "\"[empty]\"";
    }
    else if(m_id == OBJECT_T)
    {
        oss << "\"[object]\"";
    }
    else if(m_id == LIST_T)
    {
        oss << "\"[list]\"";
    }
    else
    {
        oss << "\"" << id_to_name(m_id) << "\"";
        oss << ", \"length\": " << m_num_ele;
        if(value == "")
        {
            oss << ", \"offset\": " << m_offset;
            oss << ", \"stride\": " << m_stride;
            oss << ", \"element_bytes\": " << m_ele_bytes;
        }

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
        oss << ", \"endianness\": \"" << endian_str << "\"";            
    }
    if(value != "")
    {
        oss << ", \"value\": " << value;
    }

    oss << "}";
}

};
