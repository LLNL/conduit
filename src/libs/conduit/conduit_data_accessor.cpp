// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_data_accessor.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_data_accessor.hpp"

//-----------------------------------------------------------------------------
// -- standard includes -- 
//-----------------------------------------------------------------------------
#include <algorithm>
#include <limits>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
//
// -- conduit::DataAccessor public methods --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
template <typename T> 
DataAccessor<T>::DataAccessor()
: m_data(NULL),
  m_dtype()
{}

//---------------------------------------------------------------------------//
template <typename T>
DataAccessor<T>::DataAccessor(const DataAccessor<T> &accessor)
: m_data(accessor.m_data),
  m_dtype(accessor.m_dtype)
{}


//---------------------------------------------------------------------------//
template <typename T> 
DataAccessor<T>::DataAccessor(void *data, const DataType &dtype)
: m_data(data),
  m_dtype(dtype)
{}


//---------------------------------------------------------------------------//
template <typename T> 
DataAccessor<T>::DataAccessor(const void *data, const DataType &dtype)
: m_data(const_cast<void*>(data)),
  m_dtype(dtype)
{}

//---------------------------------------------------------------------------//
template <typename T> 
DataAccessor<T>::~DataAccessor()
{} // all data is external


//---------------------------------------------------------------------------// 
///
/// Summary Stats Helpers
///
//---------------------------------------------------------------------------// 

//---------------------------------------------------------------------------// 
template <typename T>
T
DataAccessor<T>::min()  const
{
    T res = std::numeric_limits<T>::max();
    for(index_t i = 0; i < number_of_elements(); i++)
    {
        const T &val = element(i);
        if(val < res)
        {
            res = val;
        }
    }

    return res;
}

//---------------------------------------------------------------------------// 
template <typename T>
T
DataAccessor<T>::max() const
{
    T res = std::numeric_limits<T>::lowest();
    for(index_t i = 0; i < number_of_elements(); i++)
    {
        const T &val = element(i);
        if(val > res)
        {
            res = val;
        }
    }

    return res;
}


//---------------------------------------------------------------------------// 
template <typename T>
T
DataAccessor<T>::sum() const
{
    T res =0;
    for(index_t i = 0; i < number_of_elements(); i++)
    {
        const T &val = element(i);
        res += val;
    }

    return res;
}

//---------------------------------------------------------------------------// 
template <typename T>
float64
DataAccessor<T>::mean() const
{
    float64 res =0;
    for(index_t i = 0; i < number_of_elements(); i++)
    {
        const T &val = element(i);
        res += val;
    }

    res = res / float64(number_of_elements());
    return res;
}

//---------------------------------------------------------------------------// 
template <typename T>
index_t
DataAccessor<T>::count(T val) const
{
    index_t res= 0;
    for(index_t i = 0; i < number_of_elements(); i++)
    {
        if(element(i) == val)
        {
            res++;
        }
    }
    return res;
}

//---------------------------------------------------------------------------//
template <typename T> 
DataAccessor<T> &
DataAccessor<T>::operator=(const DataAccessor<T> &accessor)
{
    if(this != &accessor)
    {
        m_data  = accessor.m_data;
        m_dtype = accessor.m_dtype;
    }
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T> 
T
DataAccessor<T>::element(index_t idx) const
{
    switch(m_dtype.id())
    {
        // ints
        case DataType::INT8_ID:
            return (T)(*(int8*)(element_ptr(idx)));
        case DataType::INT16_ID: 
            return (T)(*(int16*)(element_ptr(idx)));
        case DataType::INT32_ID:
            return (T)(*(int32*)(element_ptr(idx)));
        case DataType::INT64_ID:
            return (T)(*(int64*)(element_ptr(idx)));
        // uints
        case DataType::UINT8_ID:
            return (T)(*(uint8*)(element_ptr(idx)));
        case DataType::UINT16_ID:
            return (T)(*(uint16*)(element_ptr(idx)));
        case DataType::UINT32_ID:
            return (T)(*(uint32*)(element_ptr(idx)));
        case DataType::UINT64_ID:
            return (T)(*(uint64*)(element_ptr(idx)));
        // floats 
        case DataType::FLOAT32_ID:
            return (T)(*(float32*)(element_ptr(idx)));
        case DataType::FLOAT64_ID:
            return (T)(*(float64*)(element_ptr(idx)));
    }

    // error
    CONDUIT_ERROR("DataAccessor does not support dtype: "
                  << m_dtype.name());
    return (T)0;
}



//---------------------------------------------------------------------------//
template <typename T>
void
DataAccessor<T>::set(index_t idx, T value)
{
    switch(m_dtype.id())
    {
        // ints
        case DataType::INT8_ID:
        {
            (*(int8*)(element_ptr(idx))) = static_cast<int8>(value);
            break;
        }
        case DataType::INT16_ID:
        {
            (*(int16*)(element_ptr(idx))) = static_cast<int16>(value);
            break;
        }
        case DataType::INT32_ID:
        {
            (*(int32*)(element_ptr(idx))) = static_cast<int32>(value);
            break;
        }
        case DataType::INT64_ID:
        {
            (*(int64*)(element_ptr(idx))) = static_cast<int64>(value);
            break;
        }
        // uints
        case DataType::UINT8_ID:
        {
            (*(uint8*)(element_ptr(idx))) = static_cast<uint8>(value);
            break;
        }
        case DataType::UINT16_ID:
        {
            (*(uint16*)(element_ptr(idx))) = static_cast<uint16>(value);
            break;
        }
        case DataType::UINT32_ID:
        {
            (*(uint32*)(element_ptr(idx))) = static_cast<uint32>(value);
            break;
        }
        case DataType::UINT64_ID:
        {
            (*(uint64*)(element_ptr(idx))) = static_cast<uint64>(value);
            break;
        }
        // floats
        case DataType::FLOAT32_ID:
        {
            (*(float32*)(element_ptr(idx))) = static_cast<float32>(value);
            break;
        }
        case DataType::FLOAT64_ID:
        {
            (*(float64*)(element_ptr(idx))) = static_cast<float64>(value);
            break;
        }
        default:
            // error
            CONDUIT_ERROR("DataAccessor does not support dtype: "
                          << m_dtype.name());
    }

}


//-----------------------------------------------------------------------------
//
// -- conduit::DataAccessor explicit instantiations for supported types --
//
//-----------------------------------------------------------------------------
template class DataAccessor<int8>;
template class DataAccessor<int16>;
template class DataAccessor<int32>;
template class DataAccessor<int64>;

template class DataAccessor<uint8>;
template class DataAccessor<uint16>;
template class DataAccessor<uint32>;
template class DataAccessor<uint64>;

template class DataAccessor<float32>;
template class DataAccessor<float64>;

// gap template instantiations for c-native types

// we never use 'char' directly as a type,
// so we always need to inst the char case
template class DataAccessor<char>;

#ifndef CONDUIT_USE_CHAR
template class DataAccessor<signed char>;
template class DataAccessor<unsigned char>;
#endif

#ifndef CONDUIT_USE_SHORT
template class DataAccessor<signed short>;
template class DataAccessor<unsigned short>;
#endif

#ifndef CONDUIT_USE_INT
template class DataAccessor<signed int>;
template class DataAccessor<unsigned int>;
#endif

#ifndef CONDUIT_USE_LONG
template class DataAccessor<signed long>;
template class DataAccessor<unsigned long>;
#endif

#if defined(CONDUIT_HAS_LONG_LONG) && !defined(CONDUIT_USE_LONG_LONG)
template class DataAccessor<signed long long>;
template class DataAccessor<unsigned long long>;
#endif

#ifndef CONDUIT_USE_FLOAT
template class DataAccessor<float>;
#endif

#ifndef CONDUIT_USE_DOUBLE
template class DataAccessor<double>;
#endif

#ifdef CONDUIT_USE_LONG_DOUBLE
    ltemplate class DataAccessor<long double>;
#endif


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

