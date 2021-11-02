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
DataAccessor<T>::DataAccessor()
: m_data(NULL),
  m_dtype()
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

