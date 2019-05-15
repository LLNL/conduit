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
/// file: conduit_data_array.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_DATA_ARRAY_HPP
#define CONDUIT_DATA_ARRAY_HPP

//-----------------------------------------------------------------------------
// -- conduit  includes -- 
//-----------------------------------------------------------------------------
#include "conduit_core.hpp"
#include "conduit_data_type.hpp"
#include "conduit_utils.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::DataArray --
//-----------------------------------------------------------------------------
///
/// class: conduit::DataArray
///
/// description:
///  Light weight pointer wrapper that handles addressing for ragged arrays.
///
//-----------------------------------------------------------------------------
template <typename T> 
class CONDUIT_API DataArray
{
public: 
//-----------------------------------------------------------------------------
//
// -- conduit::DataType public methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Construction and Destruction
//-----------------------------------------------------------------------------
        /// default constructor
        DataArray();
        /// copy constructor
        DataArray(const DataArray<T> &array);
        /// Access a pointer to raw data according to dtype description.
        DataArray(void *data, const DataType &dtype);
        /// Access a const pointer to raw data according to dtype description.
        DataArray(const void *data, const DataType &dtype);
        /// Destructor
       ~DataArray();

    /// Assignment operator
    DataArray<T>   &operator=(const DataArray<T> &array);

//-----------------------------------------------------------------------------
// Data and Info Access
//-----------------------------------------------------------------------------
    T              &operator[](index_t idx)
                    {return element(idx);}
    T              &operator[](index_t idx) const
                    {return element(idx);}
    
    T              &element(index_t idx);
    T              &element(index_t idx) const;

    void           *element_ptr(index_t idx)
                    {
                        return static_cast<char*>(m_data) +
                            m_dtype.element_index(idx);
                    };

    const void     *element_ptr(index_t idx) const 
                    {
                         return static_cast<char*>(m_data) +
                            m_dtype.element_index(idx);
                    };

    index_t         number_of_elements() const 
                        {return m_dtype.number_of_elements();}
    const DataType &dtype()    const 
                        { return m_dtype;} 
    void           *data_ptr() const 
                        { return m_data;}

    bool            compatible(const DataArray<T> &array) const;
    bool            diff(const DataArray<T> &array,
                         Node &info,
                         const float64 epsilon = CONDUIT_EPSILON) const;
    bool            diff_compatible(const DataArray<T> &array,
                                    Node &info,
                                    const float64 epsilon = CONDUIT_EPSILON) const;

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
    /// signed integer arrays
    void            set(const int8  *values, index_t num_elements);
    void            set(const int16 *values, index_t num_elements);
    void            set(const int32 *values, index_t num_elements);
    void            set(const int64 *values, index_t num_elements);

    /// unsigned integer arrays
    void            set(const uint8   *values, index_t num_elements);
    void            set(const uint16  *values, index_t num_elements);
    void            set(const uint32  *values, index_t num_elements);
    void            set(const uint64  *values, index_t num_elements);
    
    /// floating point arrays
    void            set(const float32 *values, index_t num_elements);
    void            set(const float64 *values, index_t num_elements);
    
    /// signed integer arrays via std::vector
    void            set(const std::vector<int8>    &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int16>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int32>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int64>   &values)
                        {set(&values[0],values.size());}

    /// unsigned integer arrays via std::vector
    void            set(const std::vector<uint8>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint16>  &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint32>  &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint64>  &values)
                        {set(&values[0],values.size());}
    
    /// floating point arrays via std::vector
    void            set(const std::vector<float32> &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<float64> &values)
                        {set(&values[0],values.size());}

    /// signed integer arrays via DataArray
    void            set(const DataArray<int8>    &values);
    void            set(const DataArray<int16>   &values);
    void            set(const DataArray<int32>   &values);
    void            set(const DataArray<int64>   &values);

    /// unsigned integer arrays via DataArray
    void            set(const DataArray<uint8>   &values);
    void            set(const DataArray<uint16>  &values);
    void            set(const DataArray<uint32>  &values);
    void            set(const DataArray<uint64>  &values);
    
    /// floating point arrays via DataArray
    void            set(const DataArray<float32>  &values);
    void            set(const DataArray<float64>  &values);

//-----------------------------------------------------------------------------
// Transforms
//-----------------------------------------------------------------------------
    std::string     to_json() const;
    void            to_json(std::ostream &os) const;
    void            compact_elements_to(uint8 *data) const;
    
//-----------------------------------------------------------------------------
// -- stdout print methods ---
//-----------------------------------------------------------------------------
    /// print a simplified json representation of the this node to std out
    void            print() const
                      {std::cout << to_json() << std::endl;}


private:

//-----------------------------------------------------------------------------
//
// -- conduit::DataArray private data members --
//
//-----------------------------------------------------------------------------
    /// holds data (always external, never allocated)
    void           *m_data;
    /// holds data description
    DataType        m_dtype;
    
};
//-----------------------------------------------------------------------------
// -- end conduit::DataArray --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- conduit::DataArray typedefs for supported array types --
//
//-----------------------------------------------------------------------------

/// Note: these are also the types we explicitly instantiate.

/// signed integer arrays
typedef DataArray<int8>     int8_array;
typedef DataArray<int16>    int16_array;
typedef DataArray<int32>    int32_array;
typedef DataArray<int64>    int64_array;

/// unsigned integer arrays
typedef DataArray<uint8>    uint8_array;
typedef DataArray<uint16>   uint16_array;
typedef DataArray<uint32>   uint32_array;
typedef DataArray<uint64>   uint64_array;

/// floating point arrays
typedef DataArray<float32>  float32_array;
typedef DataArray<float64>  float64_array;

/// native c types arrays
typedef DataArray<char>       char_array;
typedef DataArray<short>      short_array;
typedef DataArray<int>        int_array;
typedef DataArray<long>       long_array;
#ifdef CONDUIT_HAS_LONG_LONG
typedef DataArray<long long>  long_long_array;
#endif


/// signed integer arrays
typedef DataArray<signed char>       signed_char_array;
typedef DataArray<signed short>      signed_short_array;
typedef DataArray<signed int>        signed_int_array;
typedef DataArray<signed long>       signed_long_array;
#ifdef CONDUIT_HAS_LONG_LONG
typedef DataArray<signed long long>  signed_long_long_array;
#endif


/// unsigned integer arrays
typedef DataArray<unsigned char>   unsigned_char_array;
typedef DataArray<unsigned short>  unsigned_short_array;
typedef DataArray<unsigned int>    unsigned_int_array;
typedef DataArray<unsigned long>   unsigned_long_array;
#ifdef CONDUIT_HAS_LONG_LONG
typedef DataArray<unsigned long long>  unsigned_long_long_array;
#endif


/// floating point arrays
typedef DataArray<float>   float_array;
typedef DataArray<double>  double_array;
#ifdef CONDUIT_USE_LONG_DOUBLE
typedef DataArray<long double>  long_double_array;
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
