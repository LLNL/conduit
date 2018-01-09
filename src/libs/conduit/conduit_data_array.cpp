//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
/// file: conduit_data_array.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_data_array.hpp"

//-----------------------------------------------------------------------------
// -- standard includes -- 
//-----------------------------------------------------------------------------
#include <algorithm>
#include <cstring>
#include <limits>


//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "conduit_node.hpp"
#include "conduit_utils.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
//
// -- conduit::DataArray public methods --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
template <typename T> 
DataArray<T>::DataArray()
: m_data(NULL),
  m_dtype(DataType::empty())
{} 

//---------------------------------------------------------------------------//
template <typename T> 
DataArray<T>::DataArray(void *data,const DataType &dtype)
: m_data(data),
  m_dtype(dtype)
{}

//---------------------------------------------------------------------------//
template <typename T> 
DataArray<T>::DataArray(const void *data,const DataType &dtype)
: m_data(const_cast<void*>(data)),
  m_dtype(dtype)
{}


//---------------------------------------------------------------------------// 
template <typename T> 
DataArray<T>::DataArray(const DataArray<T> &array)
: m_data(array.m_data),
  m_dtype(array.m_dtype)
{}

//---------------------------------------------------------------------------//
template <typename T> 
DataArray<T>::~DataArray()
{} // all data is external

//---------------------------------------------------------------------------//
template <typename T> 
DataArray<T> &
DataArray<T>::operator=(const DataArray<T> &array)
{
    if(this != &array)
    {
        m_data  = array.m_data;
        m_dtype = array.m_dtype;
    }
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T> 
T &
DataArray<T>::element(index_t idx)
{ 
    return (*(T*)(element_ptr(idx)));
}

//---------------------------------------------------------------------------//
template <typename T> 
T &             
DataArray<T>::element(index_t idx) const 
{ 
    return (*(T*)(element_ptr(idx)));
}

//---------------------------------------------------------------------------//
template <typename T> 
bool
DataArray<T>::compatible(const DataArray<T> &array) const 
{ 
    return dtype().compatible(array.dtype());
}

//---------------------------------------------------------------------------//
template <typename T> 
bool
DataArray<T>::equals(const DataArray<T> &array, const float64 epsilon) const 
{ 
    Node info;
    return !diff(array, info, epsilon);
}

//---------------------------------------------------------------------------//
template <typename T> 
bool
DataArray<T>::diff(const DataArray<T> &array, Node &info, const float64 epsilon) const 
{ 
    bool res = false;
    info.reset();

    index_t t_nelems = number_of_elements();
    index_t o_nelems = array.number_of_elements();

    DataType info_dtype(array.dtype().id(), std::max(t_nelems, o_nelems));
    info.set(info_dtype);
    T* info_ptr = (T*)info.data_ptr();

    // TODO(JRC): This is a bit sloppy, but it feels like the only decent
    // way to signal missing values in cases where the lengths are mismatched.
    T t_fill = std::numeric_limits<T>::max();
    T o_fill = std::numeric_limits<T>::is_signed ?
        std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
    T fill_val = (t_nelems > o_nelems) ? t_fill : o_fill;

    size_t i = 0;
    for(; i < (size_t)std::min(t_nelems, o_nelems); i++)
    {
        info_ptr[i] = (*this)[i] - array[i];
        if(dtype().is_floating_point())
        {
            res |= info_ptr[i] > epsilon || info_ptr[i] < -epsilon;
        }
        else
        {
            res |= (*this)[i] != array[i];
        }
    }
    for(; i < (size_t)std::max(t_nelems, o_nelems); i++)
    {
        info_ptr[i] = fill_val;
        res |= true;
    }

    if(!res)
    {
        info.reset();
        info.set(DataType::empty());
    }

    return res;
}

//---------------------------------------------------------------------------//
template <typename T> 
std::string             
DataArray<T>::to_json() const 
{ 
    std::ostringstream oss;
    to_json(oss);
    return oss.str(); 
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::to_json(std::ostream &os) const 
{ 
    index_t nele = number_of_elements();
    if(nele > 1)
        os << "[";

    bool first=true;
    for(index_t idx = 0; idx < nele; idx++)
    {
        if(!first)
            os << ", ";
        switch(m_dtype.id())
        {
            // ints 
            case DataType::INT8_ID:
            case DataType::INT16_ID: 
            case DataType::INT32_ID:
            case DataType::INT64_ID:
            {
                 os << (int64) element(idx);
                 break;
            }
            // uints
            case DataType::UINT8_ID:
            case DataType::UINT16_ID:
            case DataType::UINT32_ID:
            case DataType::UINT64_ID:
            {
                os << (uint64) element(idx);
                break;
            }
            // floats 
            case DataType::FLOAT32_ID: 
            case DataType::FLOAT64_ID: 
            {
                std::string fs = utils::float64_to_string((float64)element(idx));
                //check for inf and nan
                // looking for 'n' covers inf and nan
                bool inf_or_nan = fs.find('n') != std::string::npos;
                
                if(inf_or_nan)
                    os << "\"";
                
                os << fs;

                if(inf_or_nan)
                    os << "\"";
                break;
            }
            default:
            {
                CONDUIT_ERROR("Leaf type \"" 
                              <<  m_dtype.name()
                              << "\"" 
                              << "is not supported in conduit::DataArray.")
            }
        }
        first=false;
    }

    if(nele > 1)
        os << "]";
}


//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const int8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const  int16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const int32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const  int64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const  uint8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const  uint16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const uint32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const uint64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const float32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const float64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// Set from DataArray
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// signed
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<int8> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<int16> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<int32> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<int64> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// unsigned
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<uint8> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<uint16> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<uint32> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<uint64> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// floating point
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<float32> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::set(const DataArray<float64> &values)
{ 
    index_t num_elems = m_dtype.number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}



//---------------------------------------------------------------------------//
template <typename T> 
void            
DataArray<T>::compact_elements_to(uint8 *data) const
{ 
    // copy all elements 
    index_t num_ele   = m_dtype.number_of_elements();
    index_t ele_bytes = DataType::default_bytes(m_dtype.id());
    uint8 *data_ptr = data;
    for(index_t i=0;i<num_ele;i++)
    {
        memcpy(data_ptr,
               element_ptr(i),
               (size_t)ele_bytes);
        data_ptr+=ele_bytes;
    }
}


//-----------------------------------------------------------------------------
//
// -- conduit::DataArray explicit instantiations for supported array types --
//
//-----------------------------------------------------------------------------
template class DataArray<int8>;
template class DataArray<int16>;
template class DataArray<int32>;
template class DataArray<int64>;

template class DataArray<uint8>;
template class DataArray<uint16>;
template class DataArray<uint32>;
template class DataArray<uint64>;

template class DataArray<float32>;
template class DataArray<float64>;

// gap template instantiations for c-native types

#ifndef CONDUIT_USE_CHAR
template class DataArray<char>;
template class DataArray<unsigned char>;
#endif

#ifndef CONDUIT_USE_SHORT
template class DataArray<short>;
template class DataArray<unsigned short>;
#endif

#ifndef CONDUIT_USE_INT
template class DataArray<int>;
template class DataArray<unsigned int>;
#endif

#ifndef CONDUIT_USE_LONG
template class DataArray<long>;
template class DataArray<unsigned long>;
#endif

#ifndef CONDUIT_USE_FLOAT
template class DataArray<float>;
#endif

#ifndef CONDUIT_USE_DOUBLE
template class DataArray<double>;
#endif



}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

