//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// file: DataType.cpp
///

#include "DataType.h"
#include "Schema.h"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- Storage for static members of conduit::DataType::Objects --
//-----------------------------------------------------------------------------
DataType DataType::Objects::m_empty(DataType::EMPTY_T);
DataType DataType::Objects::m_object(DataType::OBJECT_T);
DataType DataType::Objects::m_list(DataType::LIST_T);

//-----------------------------------------------------------------------------
// -- Storage for static members of conduit::DataType::Scalars --
//-----------------------------------------------------------------------------

/// signed integer scalars
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

/// unsigned integer scalars
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
/// floating point scalars
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

//-----------------------------------------------------------------------------
// -- conduit::DataType::Scalars members --
//-----------------------------------------------------------------------------

 /// signed integer arrays
//---------------------------------------------------------------------------//
DataType
DataType::Arrays::int8(index_t num_elements,
                       index_t offset,
                       index_t stride,
                       index_t element_bytes,
                       index_t endianness)
{
    return DataType(INT8_T,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::Arrays::int16(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(INT16_T,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType 
DataType::Arrays::int32(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(INT32_T,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::Arrays::int64(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(INT64_T,num_elements,offset,stride,element_bytes,endianness);
}

/// unsigned integer arrays
//---------------------------------------------------------------------------//
DataType
DataType::Arrays::uint8(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(UINT8_T,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::Arrays::uint16(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(UINT16_T,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType 
DataType::Arrays::uint32(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(UINT32_T,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::Arrays::uint64(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(UINT64_T,num_elements,offset,stride,element_bytes,endianness);
}

 /// floating point arrays
//---------------------------------------------------------------------------//
DataType 
DataType::Arrays::float32(index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    return DataType(FLOAT32_T,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::Arrays::float64(index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    return DataType(FLOAT64_T,num_elements,offset,stride,element_bytes,endianness);
}



//=============================================================================
//-----------------------------------------------------------------------------
//
//
// -- begin conduit::Node public methods --
//
//
//-----------------------------------------------------------------------------
//=============================================================================

//-----------------------------------------------------------------------------
// Construction and Destruction
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
DataType::DataType()
: m_id(DataType::EMPTY_T),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0),
  m_endianness(Endianness::DEFAULT_T)
{}

//---------------------------------------------------------------------------// 
DataType::DataType(const DataType& value)
: m_id(value.m_id),
  m_num_ele(value.m_num_ele),
  m_offset(value.m_offset),
  m_stride(value.m_stride),
  m_ele_bytes(value.m_ele_bytes),
  m_endianness(value.m_endianness)
{}


//---------------------------------------------------------------------------//
DataType::DataType(index_t id)
: m_id(id),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0),
  m_endianness(Endianness::DEFAULT_T)
{}

//---------------------------------------------------------------------------// 
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

//---------------------------------------------------------------------------// 
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


//---------------------------------------------------------------------------//
DataType::~DataType()
{}

//---------------------------------------------------------------------------//
void
DataType::reset()
{
    m_id = EMPTY_T;
    m_num_ele = 0;
    m_offset = 0;
    m_stride = 0;
    m_ele_bytes = 0;
    m_endianness = Endianness::DEFAULT_T;
    
}

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
void
DataType::set(const DataType& dtype)
{
    m_id = dtype.m_id;
    m_num_ele = dtype.m_num_ele;
    m_offset = dtype.m_offset;
    m_stride = dtype.m_stride;
    m_ele_bytes = dtype.m_ele_bytes;
    m_endianness = dtype.m_endianness;
}


//---------------------------------------------------------------------------// 
void
DataType::set(const std::string &dtype_name,
              index_t num_elements,
              index_t offset,
              index_t stride,
              index_t element_bytes,
              index_t endianness)
{
    m_id = name_to_id(dtype_name);
    m_num_ele = num_elements;
    m_offset = offset;
    m_stride = stride;
    m_ele_bytes = element_bytes;
    m_endianness = endianness;
}

//---------------------------------------------------------------------------// 
void
DataType::set(index_t dtype_id,
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

//-----------------------------------------------------------------------------
// Getters and info methods.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------// 
index_t
DataType::total_bytes() const
{
    // this is the memory extent
    // non compact case
    return m_stride * m_num_ele;
}

//---------------------------------------------------------------------------//
index_t
DataType::total_bytes_compact() const
{
    return default_bytes(m_id) * m_num_ele;
}
    
//---------------------------------------------------------------------------//
bool
DataType::is_compatible(const DataType& dtype) const
{
    return ( (m_id == dtype.m_id ) &&
             (m_ele_bytes == dtype.m_ele_bytes) &&
             (total_bytes() == dtype.total_bytes()));
}

//---------------------------------------------------------------------------//
bool
DataType::is_compact() const
{
    return ( (m_id != EMPTY_T) &&
             (m_id != OBJECT_T) && 
             (m_id != LIST_T) &&
             (total_bytes() == total_bytes_compact()));
}

//---------------------------------------------------------------------------//
bool
DataType::is_number() const
{
    return ( is_integer() ||
             is_float());
}

//---------------------------------------------------------------------------//
bool
DataType::is_float() const
{
    return ( (m_id == FLOAT32_T) ||
             (m_id == FLOAT64_T));
}

//---------------------------------------------------------------------------//
bool
DataType::is_integer() const
{
    return ( is_signed_integer() || 
             is_unsigned_integer());
}

//---------------------------------------------------------------------------//
bool
DataType::is_signed_integer() const
{
    return ( (m_id == INT8_T)  ||
             (m_id == INT16_T) ||
             (m_id == INT32_T) ||
             (m_id == INT64_T));
}

//---------------------------------------------------------------------------//
bool
DataType::is_unsigned_integer() const
{
    return ( (m_id == UINT8_T)  ||
             (m_id == UINT16_T) ||
             (m_id == UINT32_T) ||
             (m_id == UINT64_T));
}

//---------------------------------------------------------------------------// 
index_t     
DataType::element_index(index_t idx) const
{
    return m_offset + m_stride * idx;
}

//-----------------------------------------------------------------------------
// TypeID to string and string to TypeId
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t 
DataType::name_to_id(const std::string &dtype_name)
{
    if(dtype_name      == "[empty]") return EMPTY_T;
    else if(dtype_name == "Object")  return OBJECT_T;
    else if(dtype_name == "List")    return LIST_T;
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
    else if(dtype_name == "char8_str") return CHAR8_STR_T;
    return EMPTY_T;
}



//---------------------------------------------------------------------------//
std::string 
DataType::id_to_name(index_t dtype_id)
{
    /// container types
    if(dtype_id      == EMPTY_T)   return "[empty]";
    else if(dtype_id == OBJECT_T)  return "Object";
    else if(dtype_id == LIST_T)    return "List";
    /// signed integer types
    else if(dtype_id == INT8_T)    return "int8";
    else if(dtype_id == INT16_T)   return "int16";
    else if(dtype_id == INT32_T)   return "int32";
    else if(dtype_id == INT64_T)   return "int64";

    /// unsigned integer types
    else if(dtype_id == UINT8_T)   return "uint8";
    else if(dtype_id == UINT16_T)  return "uint16";
    else if(dtype_id == UINT32_T)  return "uint32";
    else if(dtype_id == UINT64_T)  return "uint64";

    /// floating point types
    else if(dtype_id == FLOAT32_T) return "float32";
    else if(dtype_id == FLOAT64_T) return "float64";
    /// string types
    else if(dtype_id == CHAR8_STR_T) return "char8_str";
    // default to empty
    return "[empty]";
}


//-----------------------------------------------------------------------------
// Access to simple reference data types by id or name.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------// 
DataType const &
DataType::default_dtype(index_t dtype_id)
{
   switch (dtype_id)
   {
       /// container types
       case OBJECT_T: return DataType::Objects::object();
       case LIST_T :  return DataType::Objects::list();
       
       /// signed integer types
       case INT8_T :  return DataType::Scalars::int8();
       case INT16_T : return DataType::Scalars::int16();
       case INT32_T : return DataType::Scalars::int32();
       case INT64_T : return DataType::Scalars::int64();
       
       /// unsigned integer types
       case UINT8_T :  return DataType::Scalars::uint8();
       case UINT16_T : return DataType::Scalars::uint16();
       case UINT32_T : return DataType::Scalars::uint32();
       case UINT64_T : return DataType::Scalars::uint64();

       /// floating point types
       case FLOAT32_T : return DataType::Scalars::float32();
       case FLOAT64_T : return DataType::Scalars::float64();
       /// note: there is no default dtype for char8_str
       
       /// default
       default : 
       {
           return DataType::Objects::empty();
       }
    }
}

//---------------------------------------------------------------------------// 
DataType const &
DataType::default_dtype(const std::string &name)
{
    return default_dtype(name_to_id(name));
}

//-----------------------------------------------------------------------------
// Default byte sizes for data types
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------// 
index_t
DataType::default_bytes(index_t dtype_id)
{
   switch (dtype_id)
   {
       /// signed integer types
       case INT8_T :  return sizeof(conduit::int8);
       case INT16_T : return sizeof(conduit::int16);
       case INT32_T : return sizeof(conduit::int32);
       case INT64_T : return sizeof(conduit::int64);

       /// unsigned integer types
       case UINT8_T :  return sizeof(conduit::uint8);
       case UINT16_T : return sizeof(conduit::uint16);
       case UINT32_T : return sizeof(conduit::uint32);
       case UINT64_T : return sizeof(conduit::uint64);
       
       /// floating point types
       case FLOAT32_T : return sizeof(conduit::float32);
       case FLOAT64_T : return sizeof(conduit::float64);
       /// string types
       case CHAR8_STR_T : return 1;
       /// note: there is no default bytes obj,list, or empty
       default : 
       {
           return 0;
       }
    }
}


//---------------------------------------------------------------------------// 
index_t
DataType::default_bytes(const std::string &name)
{
    return default_bytes(name_to_id(name));
}


//-----------------------------------------------------------------------------
// Transforms
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------// 
std::string 
DataType::to_json() const
{
    std::ostringstream oss;
    to_json(oss);
    return oss.str();
}

//---------------------------------------------------------------------------// 
void
DataType::to_json(std::ostringstream &oss,
                  const std::string &value) const
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

//---------------------------------------------------------------------------//
void
DataType::compact_to(DataType &dtype) const
{
     index_t ele_size =  default_bytes(m_id);
     dtype.set(m_id,
               m_num_ele, 
               0,    
               ele_size,
               ele_size,
               m_endianness);
}

//=============================================================================
//-----------------------------------------------------------------------------
//
//
// -- end conduit::DataType public methods --
//
//
//-----------------------------------------------------------------------------
//=============================================================================

};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
