//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
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

//-----------------------------------------------------------------------------
///
/// file: conduit_data_type.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_data_type.hpp"

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_utils.hpp"
#include "conduit_schema.hpp"

#ifdef CONDUIT_USE_CXX11
//-----------------------------------------------------------------------------
// Sanity checks of conduit types vs C++11 Types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Checked here to avoid header dep of C++11 in client code
//-----------------------------------------------------------------------------
#include <cstdint>
#include <type_traits>

// signed integers
static_assert(std::is_same<conduit_int8, std::int8_t>::value,
              "error: conduit_int8 != std::int8_t");

static_assert(std::is_same<conduit_int16, std::int16_t>::value,
              "error: conduit_int16 != std::int16_t");

static_assert(std::is_same<conduit_int32, std::int32_t>::value,
              "error: conduit_int32 != std::int32_t");

static_assert(std::is_same<conduit_int64, std::int64_t>::value,
              "error: conduit_int64 != std::int64_t");

// unsigned integers
static_assert(std::is_same<conduit_uint8, std::uint8_t>::value,
              "error: conduit_uint8 != std::uint8_t");

static_assert(std::is_same<conduit_uint16, std::uint16_t>::value,
              "error: conduit_uint16 != std::uint16_t");

static_assert(std::is_same<conduit_uint32, std::uint32_t>::value,
              "error: conduit_uint32 != std::uint32_t");

static_assert(std::is_same<conduit_uint64, std::uint64_t>::value,
              "error: conduit_uint64 != std::uint64_t");

#endif

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


DataType 
DataType::empty()
{
    return DataType(EMPTY_ID);
}

DataType 
DataType::object()
{
    return DataType(OBJECT_ID);
}

DataType 
DataType::list()
{
    return DataType(LIST_ID);
}



//-----------------------------------------------------------------------------
/// signed integer arrays
//---------------------------------------------------------------------------//
DataType
DataType::int8(index_t num_elements,
               index_t offset,
               index_t stride,
               index_t element_bytes,
               index_t endianness)
{
    return DataType(INT8_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::int16(index_t num_elements,
                index_t offset,
                index_t stride,
                index_t element_bytes,
                index_t endianness)
{
    return DataType(INT16_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType 
DataType::int32(index_t num_elements,
                index_t offset,
                index_t stride,
                index_t element_bytes,
                index_t endianness)
{
    return DataType(INT32_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::int64(index_t num_elements,
                index_t offset,
                index_t stride,
                index_t element_bytes,
                index_t endianness)
{
    return DataType(INT64_ID,num_elements,offset,stride,element_bytes,endianness);
}

/// unsigned integer arrays
//---------------------------------------------------------------------------//
DataType
DataType::uint8(index_t num_elements,
                index_t offset,
                index_t stride,
                index_t element_bytes,
                index_t endianness)
{
    return DataType(UINT8_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::uint16(index_t num_elements,
                 index_t offset,
                 index_t stride,
                 index_t element_bytes,
                 index_t endianness)
{
    return DataType(UINT16_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType 
DataType::uint32(index_t num_elements,
                 index_t offset,
                 index_t stride,
                 index_t element_bytes,
                 index_t endianness)
{
    return DataType(UINT32_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::uint64(index_t num_elements,
                 index_t offset,
                 index_t stride,
                 index_t element_bytes,
                 index_t endianness)
{
    return DataType(UINT64_ID,num_elements,offset,stride,element_bytes,endianness);
}

 /// floating point arrays
//---------------------------------------------------------------------------//
DataType 
DataType::float32(index_t num_elements,
                  index_t offset,
                  index_t stride,
                  index_t element_bytes,
                  index_t endianness)
{
    return DataType(FLOAT32_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::float64(index_t num_elements,
                  index_t offset,
                  index_t stride,
                  index_t element_bytes,
                  index_t endianness)
{
    return DataType(FLOAT64_ID,num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::char8_str(index_t num_elements,
                    index_t offset,
                    index_t stride,
                    index_t element_bytes,
                    index_t endianness)
{
    return DataType(CHAR8_STR_ID,
                    num_elements,
                    offset,
                    stride,
                    element_bytes,
                    endianness);
}

/// native c types
//-----------------------------------------------------------------------------
/// signed integer arrays
//---------------------------------------------------------------------------//
DataType
DataType::c_char(index_t num_elements,
                 index_t offset,
                 index_t stride,
                 index_t element_bytes,
                 index_t endianness)
{
    return DataType(CONDUIT_NATIVE_CHAR_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::c_short(index_t num_elements,
                  index_t offset,
                  index_t stride,
                  index_t element_bytes,
                  index_t endianness)
{
    return DataType(CONDUIT_NATIVE_SHORT_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType 
DataType::c_int(index_t num_elements,
                index_t offset,
                index_t stride,
                index_t element_bytes,
                index_t endianness)
{
    return DataType(CONDUIT_NATIVE_INT_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::c_long(index_t num_elements,
                 index_t offset,
                 index_t stride,
                 index_t element_bytes,
                 index_t endianness)
{
    return DataType(CONDUIT_NATIVE_LONG_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

#ifdef CONDUIT_HAS_LONG_LONG
//---------------------------------------------------------------------------//
DataType
DataType::c_long_long(index_t num_elements,
                      index_t offset,
                      index_t stride,
                      index_t element_bytes,
                      index_t endianness)
{
    return DataType(CONDUIT_NATIVE_LONG_LONG_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}
#endif

/// unsigned integer arrays
//---------------------------------------------------------------------------//
DataType
DataType::c_signed_char(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(CONDUIT_NATIVE_SIGNED_CHAR_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::c_signed_short(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(CONDUIT_NATIVE_SIGNED_SHORT_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType 
DataType::c_signed_int(index_t num_elements,
                       index_t offset,
                       index_t stride,
                       index_t element_bytes,
                       index_t endianness)
{
    return DataType(CONDUIT_NATIVE_SIGNED_INT_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::c_signed_long(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(CONDUIT_NATIVE_SIGNED_LONG_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

#ifdef CONDUIT_HAS_LONG_LONG
//---------------------------------------------------------------------------//
DataType
DataType::c_signed_long_long(index_t num_elements,
                             index_t offset,
                             index_t stride,
                             index_t element_bytes,
                             index_t endianness)
{
    return DataType(CONDUIT_NATIVE_SIGNED_LONG_LONG_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}
#endif



/// unsigned integer arrays
//---------------------------------------------------------------------------//
DataType
DataType::c_unsigned_char(index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    return DataType(CONDUIT_NATIVE_UNSIGNED_CHAR_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::c_unsigned_short(index_t num_elements,
                           index_t offset,
                           index_t stride,
                           index_t element_bytes,
                           index_t endianness)
{
    return DataType(CONDUIT_NATIVE_UNSIGNED_SHORT_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType 
DataType::c_unsigned_int(index_t num_elements,
                         index_t offset,
                         index_t stride,
                         index_t element_bytes,
                         index_t endianness)
{
    return DataType(CONDUIT_NATIVE_UNSIGNED_INT_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::c_unsigned_long(index_t num_elements,
                          index_t offset,
                          index_t stride,
                          index_t element_bytes,
                          index_t endianness)
{
    return DataType(CONDUIT_NATIVE_UNSIGNED_LONG_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

#ifdef CONDUIT_HAS_LONG_LONG
//---------------------------------------------------------------------------//
DataType
DataType::c_unsigned_long_long(index_t num_elements,
                               index_t offset,
                               index_t stride,
                               index_t element_bytes,
                               index_t endianness)
{
    return DataType(CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}
#endif

 /// floating point arrays
//---------------------------------------------------------------------------//
DataType 
DataType::c_float(index_t num_elements,
                  index_t offset,
                  index_t stride,
                  index_t element_bytes,
                  index_t endianness)
{
    return DataType(CONDUIT_NATIVE_FLOAT_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

//---------------------------------------------------------------------------//
DataType
DataType::c_double(index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness)
{
    return DataType(CONDUIT_NATIVE_DOUBLE_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}

#ifdef CONDUIT_USE_LONG_DOUBLE
//---------------------------------------------------------------------------//
DataType
DataType::c_long_double(index_t num_elements,
                        index_t offset,
                        index_t stride,
                        index_t element_bytes,
                        index_t endianness)
{
    return DataType(CONDUIT_NATIVE_LONG_DOUBLE_ID,
                    num_elements,offset,stride,element_bytes,endianness);
}
#endif


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
: m_id(DataType::EMPTY_ID),
  m_num_ele(0),
  m_offset(0),
  m_stride(0),
  m_ele_bytes(0),
  m_endianness(Endianness::DEFAULT_ID)
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
DataType::DataType(index_t id, index_t num_elements)
: m_id(id),
  m_num_ele(num_elements),
  m_offset(0),
  m_stride(DataType::default_bytes(id)),
  m_ele_bytes(DataType::default_bytes(id)),
  m_endianness(Endianness::DEFAULT_ID)
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
    m_id = EMPTY_ID;
    m_num_ele = 0;
    m_offset = 0;
    m_stride = 0;
    m_ele_bytes = 0;
    m_endianness = Endianness::DEFAULT_ID;
    
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
DataType::strided_bytes() const
{
    // this is the memory extent excluding the offset
    return m_stride * (m_num_ele -1) + m_ele_bytes;
}


//---------------------------------------------------------------------------//
index_t
DataType::bytes_compact() const
{
    return default_bytes(m_id) * m_num_ele;
}

//---------------------------------------------------------------------------//
index_t
DataType::spanned_bytes() const
{
    return m_offset + m_stride * (m_num_ele -1) + m_ele_bytes;
}

//---------------------------------------------------------------------------//
bool
DataType::compatible(const DataType& dtype) const
{
    return ( (m_id == dtype.m_id ) &&
             (m_ele_bytes == dtype.m_ele_bytes) &&
             (m_num_ele >= dtype.m_num_ele) );
}

//---------------------------------------------------------------------------//
bool
DataType::equals(const DataType& dtype) const
{
    return ( (m_id == dtype.m_id ) &&
             (m_num_ele   == dtype.m_num_ele) &&
             (m_offset    == dtype.m_offset) &&
             (m_ele_bytes == dtype.m_ele_bytes) &&
             (m_endianness == dtype.m_endianness));
}


//---------------------------------------------------------------------------//
bool
DataType::is_compact() const
{
    return ( (m_id != EMPTY_ID) &&
             (m_id != OBJECT_ID) && 
             (m_id != LIST_ID) &&
             (spanned_bytes() == bytes_compact()));
}

//---------------------------------------------------------------------------//
bool
DataType::is_empty() const
{
    return m_id == EMPTY_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_object() const
{
    return m_id == OBJECT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_list() const
{
    return m_id == LIST_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_number() const
{
    return ( is_integer() ||
             is_floating_point());
}

//---------------------------------------------------------------------------//
bool
DataType::is_floating_point() const
{
    return ( (m_id == FLOAT32_ID) ||
             (m_id == FLOAT64_ID));
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
    return ( (m_id == INT8_ID)  ||
             (m_id == INT16_ID) ||
             (m_id == INT32_ID) ||
             (m_id == INT64_ID));
}

//---------------------------------------------------------------------------//
bool
DataType::is_unsigned_integer() const
{
    return ( (m_id == UINT8_ID)  ||
             (m_id == UINT16_ID) ||
             (m_id == UINT32_ID) ||
             (m_id == UINT64_ID));
}

//---------------------------------------------------------------------------//
bool
DataType::is_int8() const
{
    return m_id == INT8_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_int16() const
{
    return m_id == INT16_ID;
}


//---------------------------------------------------------------------------//
bool
DataType::is_int32() const
{
    return m_id == INT32_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_int64() const
{
    return m_id == INT64_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_uint8() const
{
    return m_id == UINT8_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_uint16() const
{
    return m_id == UINT16_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_uint32() const
{
    return m_id == UINT32_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_uint64() const
{
    return m_id == UINT64_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_float32() const
{
    return m_id == FLOAT32_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_float64() const
{
    return m_id == FLOAT64_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_char() const
{
    return m_id == CONDUIT_NATIVE_CHAR_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_short() const
{
    return m_id == CONDUIT_NATIVE_SHORT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_int() const
{
    return m_id == CONDUIT_NATIVE_INT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_long() const
{
    return m_id == CONDUIT_NATIVE_LONG_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_long_long() const
{
#ifdef CONDUIT_HAS_LONG_LONG
    return m_id == CONDUIT_NATIVE_LONG_LONG_ID;
#else
    return false;
#endif
}


//---------------------------------------------------------------------------//
bool
DataType::is_signed_char() const
{
    return m_id == CONDUIT_NATIVE_SIGNED_CHAR_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_signed_short() const
{
    return m_id == CONDUIT_NATIVE_SIGNED_SHORT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_signed_int() const
{
    return m_id == CONDUIT_NATIVE_SIGNED_INT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_signed_long() const
{
    return m_id == CONDUIT_NATIVE_SIGNED_LONG_ID;
}


//---------------------------------------------------------------------------//
bool
DataType::is_signed_long_long() const
{
#ifdef CONDUIT_HAS_LONG_LONG
    return m_id == CONDUIT_NATIVE_SIGNED_LONG_LONG_ID;
#else
    return false;
#endif
}


//---------------------------------------------------------------------------//
bool
DataType::is_unsigned_char() const
{
    return m_id == CONDUIT_NATIVE_UNSIGNED_CHAR_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_unsigned_short() const
{
    return m_id == CONDUIT_NATIVE_UNSIGNED_SHORT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_unsigned_int() const
{
    return m_id == CONDUIT_NATIVE_UNSIGNED_INT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_unsigned_long() const
{
    return m_id == CONDUIT_NATIVE_UNSIGNED_LONG_ID;
}


//---------------------------------------------------------------------------//
bool
DataType::is_unsigned_long_long() const
{
#ifdef CONDUIT_HAS_LONG_LONG
    return m_id == CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID;
#else
    return false;
#endif
}


//---------------------------------------------------------------------------//
bool
DataType::is_float() const
{
    return m_id == CONDUIT_NATIVE_FLOAT_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_double() const
{
    return m_id == CONDUIT_NATIVE_DOUBLE_ID;
}


//---------------------------------------------------------------------------//
bool
DataType::is_long_double() const
{
#ifdef CONDUIT_USE_LONG_DOUBLE
    return m_id == CONDUIT_NATIVE_LONG_DOUBLE_ID;
#else
    return false;
#endif
}


//---------------------------------------------------------------------------//
bool
DataType::is_string() const
{
    // we only support one string type
    return is_char8_str();
}

//---------------------------------------------------------------------------//
bool
DataType::is_char8_str() const
{
    return m_id == CHAR8_STR_ID;
}

//---------------------------------------------------------------------------//
bool
DataType::is_little_endian() const
{
    return ( (m_endianness == Endianness::LITTLE_ID) ||
             (m_endianness ==  Endianness::DEFAULT_ID 
                && Endianness::machine_is_little_endian())
            );
}

//---------------------------------------------------------------------------//
bool
DataType::is_big_endian() const
{
    return ( (m_endianness == Endianness::BIG_ID) ||
             (m_endianness ==  Endianness::DEFAULT_ID 
                && Endianness::machine_is_big_endian())
            );
}

//---------------------------------------------------------------------------//
bool
DataType::endianness_matches_machine() const
{
    return ( (m_endianness ==  Endianness::DEFAULT_ID)   ||
             (m_endianness == Endianness::BIG_ID && 
                    Endianness::machine_is_big_endian()) ||
             (m_endianness == Endianness::LITTLE_ID && 
                    Endianness::machine_is_little_endian()) );
}

//---------------------------------------------------------------------------// 
index_t
DataType::element_index(index_t idx) const
{
    /// TODO: This will be an expensive check, placed this in 
    /// to help us ferret out some places were we are creating default 
    /// datatypes that have stride == 0.

    if(idx > 0 && m_stride == 0)
    {
        CONDUIT_WARN("Node index calculation with with stride = 0");
    }
    
    return m_offset + m_stride * idx;
}

//-----------------------------------------------------------------------------
// TypeID to string and string to TypeId
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t 
DataType::name_to_id(const std::string &dtype_name)
{
    if(dtype_name      == "empty")   return EMPTY_ID;
    else if(dtype_name == "object")  return OBJECT_ID;
    else if(dtype_name == "list")    return LIST_ID;
    else if(dtype_name == "int8")    return INT8_ID;
    else if(dtype_name == "int16")   return INT16_ID;
    else if(dtype_name == "int32")   return INT32_ID;
    else if(dtype_name == "int64")   return INT64_ID;
    else if(dtype_name == "uint8")   return UINT8_ID;
    else if(dtype_name == "uint16")  return UINT16_ID;
    else if(dtype_name == "uint32")  return UINT32_ID;
    else if(dtype_name == "uint64")  return UINT64_ID;
    else if(dtype_name == "float32") return FLOAT32_ID;
    else if(dtype_name == "float64") return FLOAT64_ID;
    else if(dtype_name == "char8_str") return CHAR8_STR_ID;
    return EMPTY_ID;
}



//---------------------------------------------------------------------------//
std::string 
DataType::id_to_name(index_t dtype_id)
{
    /// container types
    if(dtype_id      == EMPTY_ID)   return "empty";
    else if(dtype_id == OBJECT_ID)  return "object";
    else if(dtype_id == LIST_ID)    return "list";

    /// signed integer types
    else if(dtype_id == INT8_ID)    return "int8";
    else if(dtype_id == INT16_ID)   return "int16";
    else if(dtype_id == INT32_ID)   return "int32";
    else if(dtype_id == INT64_ID)   return "int64";

    /// unsigned integer types
    else if(dtype_id == UINT8_ID)   return "uint8";
    else if(dtype_id == UINT16_ID)  return "uint16";
    else if(dtype_id == UINT32_ID)  return "uint32";
    else if(dtype_id == UINT64_ID)  return "uint64";

    /// floating point types
    else if(dtype_id == FLOAT32_ID) return "float32";
    else if(dtype_id == FLOAT64_ID) return "float64";
    /// string types
    else if(dtype_id == CHAR8_STR_ID) return "char8_str";
    // default to empty
    return "empty";
}

//-----------------------------------------------------------------------------
// TypeID to string and string to TypeId
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
index_t 
DataType::c_type_name_to_id(const std::string &dtype_name)
{
    if(dtype_name == "char")
        return CONDUIT_NATIVE_CHAR_ID;
    else if(dtype_name == "short")
        return CONDUIT_NATIVE_SHORT_ID;
    else if(dtype_name == "int")
        return CONDUIT_NATIVE_INT_ID;
    else if(dtype_name == "long")
        return CONDUIT_NATIVE_LONG_ID;
#ifdef CONDUIT_HAS_LONG_LONG
    else if(dtype_name == "long long")
        return CONDUIT_NATIVE_LONG_LONG_ID;
#endif
    else if(dtype_name == "signed char")
        return CONDUIT_NATIVE_SIGNED_CHAR_ID;
    else if(dtype_name == "signed short")
        return CONDUIT_NATIVE_SIGNED_SHORT_ID;
    else if(dtype_name == "signed int")
        return CONDUIT_NATIVE_SIGNED_INT_ID;
    else if(dtype_name == "signed long")
        return CONDUIT_NATIVE_SIGNED_LONG_ID;
#ifdef CONDUIT_HAS_LONG_LONG
    else if(dtype_name == "signed long long")
        return CONDUIT_NATIVE_SIGNED_LONG_LONG_ID;
#endif
    else if(dtype_name == "unsigned char")
        return CONDUIT_NATIVE_UNSIGNED_CHAR_ID;
    else if(dtype_name == "unsigned short")
        return CONDUIT_NATIVE_UNSIGNED_SHORT_ID;
    else if(dtype_name == "unsigned int")
        return CONDUIT_NATIVE_UNSIGNED_INT_ID;
    else if(dtype_name == "unsigned long")
        return CONDUIT_NATIVE_UNSIGNED_LONG_ID;
#ifdef CONDUIT_HAS_LONG_LONG
    else if(dtype_name == "unsigned long long")
        return CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID;
#endif
    else if(dtype_name == "float")
        return CONDUIT_NATIVE_FLOAT_ID;
    else if(dtype_name == "double")
        return CONDUIT_NATIVE_DOUBLE_ID;
#ifdef CONDUIT_USE_LONG_DOUBLE
    else if(dtype_name == "long double")
        return CONDUIT_NATIVE_LONG_DOUBLE_ID;
#endif
    else if(dtype_name == "char8_str")
        return CHAR8_STR_ID;
    return EMPTY_ID;
}


//-----------------------------------------------------------------------------
// Access to simple reference data types by id or name.
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------// 
DataType
DataType::default_dtype(index_t dtype_id)
{
   switch (dtype_id)
   {
       /// container types
       case OBJECT_ID: return DataType::object();
       case LIST_ID :  return DataType::list();
       
       /// signed integer types
       case INT8_ID :  return DataType::int8();
       case INT16_ID : return DataType::int16();
       case INT32_ID : return DataType::int32();
       case INT64_ID : return DataType::int64();
       
       /// unsigned integer types
       case UINT8_ID :  return DataType::uint8();
       case UINT16_ID : return DataType::uint16();
       case UINT32_ID : return DataType::uint32();
       case UINT64_ID : return DataType::uint64();

       /// floating point types
       case FLOAT32_ID : return DataType::float32();
       case FLOAT64_ID : return DataType::float64();
       /// note: there is no default dtype for char8_str
       
       /// default
       default : 
       {
           return DataType::empty();
       }
    }
}

//---------------------------------------------------------------------------// 
DataType
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
       case INT8_ID :  return sizeof(conduit::int8);
       case INT16_ID : return sizeof(conduit::int16);
       case INT32_ID : return sizeof(conduit::int32);
       case INT64_ID : return sizeof(conduit::int64);

       /// unsigned integer types
       case UINT8_ID :  return sizeof(conduit::uint8);
       case UINT16_ID : return sizeof(conduit::uint16);
       case UINT32_ID : return sizeof(conduit::uint32);
       case UINT64_ID : return sizeof(conduit::uint64);
       
       /// floating point types
       case FLOAT32_ID : return sizeof(conduit::float32);
       case FLOAT64_ID : return sizeof(conduit::float64);
       /// string types
       case CHAR8_STR_ID : return 1;
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
    to_json_stream(oss);
    return oss.str();
}

//---------------------------------------------------------------------------// 
void
DataType::to_json_stream(std::ostream &os) const
{
    os << "{\"dtype\":" << "\"" << id_to_name(m_id) << "\"";

    if(is_number() || is_string())
    {
        os << ", \"number_of_elements\": " << m_num_ele;
        os << ", \"offset\": " << m_offset;
        os << ", \"stride\": " << m_stride;
        os << ", \"element_bytes\": " << m_ele_bytes;

        std::string endian_str;
        if(m_endianness == Endianness::DEFAULT_ID)
        {
            // find this machine's actual endianness
            endian_str = Endianness::id_to_name(Endianness::machine_default());
        }
        else
        {
            endian_str = Endianness::id_to_name(m_endianness);
        }
        os << ", \"endianness\": \"" << endian_str << "\"";
    }

    os << "}";
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

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
