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
/// file: conduit_data_type.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_DATA_TYPE_HPP
#define CONDUIT_DATA_TYPE_HPP

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <vector>
#include <string>
#include <sstream>

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "conduit_core.hpp"
#include "conduit_endianness.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- forward declarations required for conduit::Node --
//-----------------------------------------------------------------------------
class Schema;
class Node;

//-----------------------------------------------------------------------------
// -- begin conduit::DataType --
//-----------------------------------------------------------------------------
///
/// class: conduit::DataType
///
/// description:
///  DataType is used to describe a single entry in a Schema or Node Hierarchy.
///
//-----------------------------------------------------------------------------
class CONDUIT_API DataType
{
public:
//-----------------------------------------------------------------------------
/// TypeID is an Enumeration used to describe the type cases supported
///  by conduit:
//-----------------------------------------------------------------------------
    typedef enum
    {
        EMPTY_ID     = CONDUIT_EMPTY_ID,     // empty (default type)
        OBJECT_ID    = CONDUIT_OBJECT_ID,    // object
        LIST_ID      = CONDUIT_LIST_ID,      // list
        INT8_ID      = CONDUIT_INT8_ID,      // int8 and int8_array
        INT16_ID     = CONDUIT_INT16_ID,     // int16 and int16_array
        INT32_ID     = CONDUIT_INT32_ID,     // int32 and int32_array
        INT64_ID     = CONDUIT_INT64_ID,     // int64 and int64_array
        UINT8_ID     = CONDUIT_UINT8_ID,     // int8 and int8_array
        UINT16_ID    = CONDUIT_UINT16_ID,    // uint16 and uint16_array
        UINT32_ID    = CONDUIT_UINT32_ID,    // uint32 and uint32_array
        UINT64_ID    = CONDUIT_UINT64_ID,    // uint64 and uint64_array
        FLOAT32_ID   = CONDUIT_FLOAT32_ID,   // float32 and float32_array
        FLOAT64_ID   = CONDUIT_FLOAT64_ID,   // float64 and float64_array
        CHAR8_STR_ID = CONDUIT_CHAR8_STR_ID  // char8 string (incore c-string)
    } TypeID;

//-----------------------------------------------------------------------------
// -- begin conduit::DataType Objects Constructor Helpers --
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
///
/// class: conduit::DataType:Objects
///
/// description:
///  Reference DataType instances for "object" types.
///
//-----------------------------------------------------------------------------

    static DataType empty();
    static DataType object();
    static DataType list();
    
//-----------------------------------------------------------------------------
// -- end conduit::DataType Objects Constructor Helpers --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::DataType Leaf Constructor Helpers --
//-----------------------------------------------------------------------------
    /// signed integer arrays
    static DataType int8(index_t num_elements=1,
                         index_t offset = 0,
                         index_t stride = sizeof(conduit::int8),
                         index_t element_bytes = sizeof(conduit::int8),
                         index_t endianness = Endianness::DEFAULT_ID);

    static DataType int16(index_t num_elements=1,
                          index_t offset = 0,
                          index_t stride = sizeof(conduit::int16),
                          index_t element_bytes = sizeof(conduit::int16),
                          index_t endianness = Endianness::DEFAULT_ID);

    static DataType int32(index_t num_elements=1,
                          index_t offset = 0,
                          index_t stride = sizeof(conduit::int32),
                          index_t element_bytes = sizeof(conduit::int32),
                          index_t endianness = Endianness::DEFAULT_ID);

    static DataType int64(index_t num_elements=1,
                          index_t offset = 0,
                          index_t stride = sizeof(conduit::int64),
                          index_t element_bytes = sizeof(conduit::int64),
                          index_t endianness = Endianness::DEFAULT_ID);

    /// unsigned integer arrays
    static DataType uint8(index_t num_elements=1,
                          index_t offset = 0,
                          index_t stride = sizeof(conduit::uint8),
                          index_t element_bytes = sizeof(conduit::uint8),
                          index_t endianness = Endianness::DEFAULT_ID);

    static DataType uint16(index_t num_elements=1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::uint16),
                           index_t element_bytes = sizeof(conduit::uint16),
                           index_t endianness = Endianness::DEFAULT_ID);

    static DataType uint32(index_t num_elements=1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::uint32),
                           index_t element_bytes = sizeof(conduit::uint32),
                           index_t endianness = Endianness::DEFAULT_ID);

    static DataType uint64(index_t num_elements=1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::uint64),
                           index_t element_bytes = sizeof(conduit::uint64),
                           index_t endianness = Endianness::DEFAULT_ID);

    /// floating point arrays
    static DataType float32(index_t num_elements=1,
                            index_t offset = 0,
                            index_t stride = sizeof(conduit::float32),
                            index_t element_bytes=sizeof(conduit::float32),
                            index_t endianness = Endianness::DEFAULT_ID);

    static DataType float64(index_t num_elements=1,
                            index_t offset = 0,
                            index_t stride = sizeof(conduit::float64),
                            index_t element_bytes=sizeof(conduit::float64),
                            index_t endianness = Endianness::DEFAULT_ID);

    static DataType char8_str(index_t num_elements=1,
                              index_t offset = 0,
                              index_t stride = 1,
                              index_t element_bytes=1,
                              index_t endianness = Endianness::DEFAULT_ID);


//-----------------------------------------------------------------------------
// -- end conduit::DataType Leaf Constructor Helpers --
//-----------------------------------------------------------------------------
                            
//-----------------------------------------------------------------------------
// -- begin conduit::DataType C Native Leaf Constructor Helpers --
//-----------------------------------------------------------------------------
    /// signed integer arrays
    static DataType c_char(index_t num_elements=1,
                           index_t offset = 0,
                           index_t stride = sizeof(CONDUIT_NATIVE_CHAR),
                           index_t element_bytes = sizeof(CONDUIT_NATIVE_CHAR),
                           index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_short(index_t num_elements=1,
                            index_t offset = 0,
                            index_t stride = sizeof(CONDUIT_NATIVE_SHORT),
                            index_t element_bytes = sizeof(CONDUIT_NATIVE_SHORT),
                            index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_int(index_t num_elements=1,
                          index_t offset = 0,
                          index_t stride = sizeof(CONDUIT_NATIVE_INT),
                          index_t element_bytes = sizeof(CONDUIT_NATIVE_INT),
                          index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_long(index_t num_elements=1,
                           index_t offset = 0,
                           index_t stride = sizeof(CONDUIT_NATIVE_LONG),
                           index_t element_bytes = sizeof(CONDUIT_NATIVE_LONG),
                           index_t endianness = Endianness::DEFAULT_ID);

#ifdef CONDUIT_HAS_LONG_LONG
    static DataType c_long_long(index_t num_elements=1,
                                index_t offset = 0,
                                index_t stride = sizeof(CONDUIT_NATIVE_LONG_LONG),
                                index_t element_bytes = sizeof(CONDUIT_NATIVE_LONG_LONG),
                                index_t endianness = Endianness::DEFAULT_ID);
#endif

    /// signed integer arrays
    static DataType c_signed_char(index_t num_elements=1,
                                  index_t offset = 0,
                                  index_t stride = sizeof(CONDUIT_NATIVE_SIGNED_CHAR),
                                  index_t element_bytes =  sizeof(CONDUIT_NATIVE_SIGNED_CHAR),
                                  index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_signed_short(index_t num_elements=1,
                                   index_t offset = 0,
                                   index_t stride = sizeof(CONDUIT_NATIVE_SIGNED_SHORT),
                                   index_t element_bytes = sizeof(CONDUIT_NATIVE_SIGNED_SHORT),
                                   index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_signed_int(index_t num_elements=1,
                                 index_t offset = 0,
                                 index_t stride = sizeof(CONDUIT_NATIVE_SIGNED_INT),
                                 index_t element_bytes = sizeof(CONDUIT_NATIVE_SIGNED_INT),
                                 index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_signed_long(index_t num_elements=1,
                                  index_t offset = 0,
                                  index_t stride = sizeof(CONDUIT_NATIVE_SIGNED_LONG),
                                  index_t element_bytes = sizeof(CONDUIT_NATIVE_SIGNED_LONG),
                                   index_t endianness = Endianness::DEFAULT_ID);

#ifdef CONDUIT_HAS_LONG_LONG
    static DataType c_signed_long_long(index_t num_elements=1,
                                       index_t offset = 0,
                                       index_t stride = sizeof(CONDUIT_NATIVE_SIGNED_LONG_LONG),
                                       index_t element_bytes = sizeof(CONDUIT_NATIVE_SIGNED_LONG_LONG),
                                       index_t endianness = Endianness::DEFAULT_ID);
#endif

    /// unsigned integer arrays
    static DataType c_unsigned_char(index_t num_elements=1,
                                    index_t offset = 0,
                                    index_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_CHAR),
                                    index_t element_bytes =  sizeof(CONDUIT_NATIVE_UNSIGNED_CHAR),
                                    index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_unsigned_short(index_t num_elements=1,
                                     index_t offset = 0,
                                     index_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_SHORT),
                                     index_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_SHORT),
                                     index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_unsigned_int(index_t num_elements=1,
                                   index_t offset = 0,
                                   index_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_INT),
                                   index_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_INT),
                                   index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_unsigned_long(index_t num_elements=1,
                                    index_t offset = 0,
                                    index_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_LONG),
                                    index_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_LONG),
                                    index_t endianness = Endianness::DEFAULT_ID);

#ifdef CONDUIT_HAS_LONG_LONG
    static DataType c_unsigned_long_long(index_t num_elements=1,
                                         index_t offset = 0,
                                         index_t stride = sizeof(CONDUIT_NATIVE_UNSIGNED_LONG_LONG),
                                         index_t element_bytes = sizeof(CONDUIT_NATIVE_UNSIGNED_LONG_LONG),
                                         index_t endianness = Endianness::DEFAULT_ID);
#endif

    /// floating point arrays
    static DataType c_float(index_t num_elements=1,
                            index_t offset = 0,
                            index_t stride = sizeof(CONDUIT_NATIVE_FLOAT),
                            index_t element_bytes=sizeof(CONDUIT_NATIVE_FLOAT),
                            index_t endianness = Endianness::DEFAULT_ID);

    static DataType c_double(index_t num_elements=1,
                             index_t offset = 0,
                             index_t stride = sizeof(CONDUIT_NATIVE_DOUBLE),
                             index_t element_bytes=sizeof(CONDUIT_NATIVE_DOUBLE),
                             index_t endianness = Endianness::DEFAULT_ID);

#ifdef CONDUIT_USE_LONG_DOUBLE
    static DataType c_long_double(index_t num_elements=1,
                             index_t offset = 0,
                             index_t stride = sizeof(CONDUIT_NATIVE_LONG_DOUBLE),
                             index_t element_bytes=sizeof(CONDUIT_NATIVE_LONG_DOUBLE),
                             index_t endianness = Endianness::DEFAULT_ID);
#endif


//-----------------------------------------------------------------------------
// -- begin conduit::DataType C Native Leaf Constructor Helpers 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- conduit::DataType public methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Construction and Destruction
//-----------------------------------------------------------------------------

    /// standard constructor
    DataType();
    /// copy constructor
    DataType(const DataType& type);
    /// construct simplest dtype for given type id
    explicit DataType(index_t id, index_t num_elements=0);

    /// construct from full details, given a data type name
    DataType(const std::string &dtype_name,
             index_t num_elements,
             index_t offset,
             index_t stride,
             index_t element_bytes,
             index_t endianness);

    /// construct from full details, given a data type id
    DataType(index_t dtype_id,
             index_t num_elements,
             index_t offset,
             index_t stride,
             index_t element_bytes,
             index_t endianness);

    /// destructor
   ~DataType();

   /// return a data type to the default (empty) state
   void  reset();
   
//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
    void       set(const DataType& type);
    
    void       set(const std::string &dtype_name,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness);    

    void       set(index_t dtype_id,
                   index_t num_elements,
                   index_t offset,
                   index_t stride,
                   index_t element_bytes,
                   index_t endianness);
    
    void       set_id(index_t dtype_id)
                    { m_id = dtype_id;}
                   
    void       set_number_of_elements(index_t v)
                    { m_num_ele = v;}
    void       set_offset(index_t v)
                    { m_offset = v;}
    void       set_stride(index_t v)
                    { m_stride = v;}
    void       set_element_bytes(index_t v)
                    { m_ele_bytes = v;}
    void       set_endianness(index_t v)
                    { m_endianness = v;}

//-----------------------------------------------------------------------------
// Getters and info methods.
//-----------------------------------------------------------------------------
    index_t     id()    const { return m_id;}
    std::string name()  const { return id_to_name(m_id);}

    index_t     number_of_elements()  const { return m_num_ele;}
    index_t     offset()              const { return m_offset;}
    index_t     stride()              const { return m_stride;}
    index_t     element_bytes()       const { return m_ele_bytes;}
    index_t     endianness()          const { return m_endianness;}
    index_t     element_index(index_t idx) const;

    /// strided bytes = stride() * (number_of_elements() -1) + element_bytes()
    index_t     strided_bytes() const;
    // bytes compact = number_of_elements() * element_bytes()
    index_t     bytes_compact() const;
    /// spanned bytes = strided_bytes() + offet()
    index_t     spanned_bytes() const;

    bool        is_compact() const;
    bool        compatible(const DataType& type) const;
    bool        equals(const DataType& type) const;

    bool        is_empty()            const;
    bool        is_object()           const;
    bool        is_list()             const;

    bool        is_number()           const;
    bool        is_floating_point()   const;
    bool        is_integer()          const;
    bool        is_signed_integer()   const;
    bool        is_unsigned_integer() const;
    
    bool        is_int8()             const;
    bool        is_int16()            const;
    bool        is_int32()            const;
    bool        is_int64()            const;

    bool        is_uint8()            const;
    bool        is_uint16()           const;
    bool        is_uint32()           const;
    bool        is_uint64()           const;

    bool        is_float32()          const;
    bool        is_float64()          const;
    bool        is_index_t()          const;

    // native c types
    bool        is_char()             const;
    bool        is_short()            const;
    bool        is_int()              const;
    bool        is_long()             const;
    /// note: is_long_long() always returns false if conduit is not
    /// using long long to fill its support for bitwidth style types
    bool        is_long_long()      const;

    // signed c types
    bool        is_signed_char()    const;
    bool        is_signed_short()   const;
    bool        is_signed_int()     const;
    bool        is_signed_long()    const;
    /// note: is_signed_long_long() always returns false if conduit is not
    /// using long long to fill its support for bitwidth style types
    bool        is_signed_long_long() const;

    // unsigned c types
    bool        is_unsigned_char()    const;
    bool        is_unsigned_short()   const;
    bool        is_unsigned_int()     const;
    bool        is_unsigned_long()    const;
    /// note: is_unsigned_long_long() always returns false if conduit is not
    /// using long long to fill its support for bitwidth style types
    bool        is_unsigned_long_long() const;

    // floating point c types
    bool        is_float()            const;
    bool        is_double()           const;
    /// note: is_long_double() always returns false if conduit is not using 
    /// long double to fill its support for bitwidth style types
    bool        is_long_double()      const;

    // strings
    bool        is_string()           const;
    bool        is_char8_str()        const;

    // endianness
    bool        is_little_endian()    const;
    bool        is_big_endian()       const;
    bool        endianness_matches_machine() const;

//-----------------------------------------------------------------------------
// Helpers to convert TypeID Enum Values to human readable strings and 
// vice versa.
//-----------------------------------------------------------------------------
    static index_t          name_to_id(const std::string &name);
    static std::string      id_to_name(index_t dtype);
    static index_t          c_type_name_to_id(const std::string &name);

//-----------------------------------------------------------------------------
// Access to simple reference data types by id or name.
//-----------------------------------------------------------------------------

    static DataType default_dtype(index_t dtype_id);
    static DataType default_dtype(const std::string &name);

//-----------------------------------------------------------------------------
// Return the default number of bytes used in a given type (from a type id, or
// string)
//-----------------------------------------------------------------------------
    static index_t          default_bytes(index_t dtype_id);
    static index_t          default_bytes(const std::string &name);


//-----------------------------------------------------------------------------
// Transforms
//-----------------------------------------------------------------------------
    std::string         to_json() const;  
    void                to_json_stream(std::ostream &os) const;

    void                compact_to(DataType &dtype) const;


private:
//-----------------------------------------------------------------------------
//
// -- conduit::DataType private data members --
//
//-----------------------------------------------------------------------------
    index_t   m_id;         /// for dtype enum value
    index_t   m_num_ele;    /// number of elements
    index_t   m_offset;     /// bytes to start of array
    index_t   m_stride;     /// bytes between start of current and start of next
    index_t   m_ele_bytes;  /// bytes per element
    index_t   m_endianness; /// endianness of elements

};
//-----------------------------------------------------------------------------
// -- end conduit::DataType --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
