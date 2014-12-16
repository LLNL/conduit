/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: DataArray.cpp
///

#include "DataArray.h"
#include <cstring>

namespace conduit
{


//============================================
// DataArray
//============================================


//============================================
template <typename T> 
DataArray<T>::DataArray(void *data,const DataType &dtype)
: m_data(data),
  m_dtype(dtype)
{}

//============================================
template <typename T> 
DataArray<T>::DataArray(const void *data,const DataType &dtype)
: m_data(const_cast<void*>(data)),
  m_dtype(dtype)
{}


//============================================ 
template <typename T> 
DataArray<T>::DataArray(const DataArray<T> &array)
: m_data(array.m_data),
  m_dtype(array.m_dtype)
{}

//============================================
template <typename T> 
DataArray<T>::~DataArray()
{} // all data is external

//============================================
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

//============================================
template <typename T> 
T &
DataArray<T>::element(index_t idx)
{ 
    //    std::cout << "[" << idx << "] = idx "
    //              << m_dtype.element_index(idx) << std::endl;
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}

//============================================
template <typename T> 
T &             
DataArray<T>::element(index_t idx) const 
{ 
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}

//============================================
template <typename T> 
std::string             
DataArray<T>::to_json() const 
{ 
    std::ostringstream oss;
    to_json(oss);
    return oss.str(); 
}

//============================================
template <typename T> 
void            
DataArray<T>::to_json(std::ostringstream &oss) const 
{ 
    index_t nele = number_of_elements();
    if(nele > 1)
        oss << "[";

    bool first=true;
    for(index_t idx = 0; idx < nele; idx++)
    {
        if(!first)
            oss << ", ";
        switch(m_dtype.id())
        {
            /// TODO: This could be orged better
            /* ints */
            case DataType::INT8_T:  oss << (int64) element(idx); break;
            case DataType::INT16_T: oss << (int64) element(idx); break;
            case DataType::INT32_T: oss << (int64) element(idx); break;
            case DataType::INT64_T: oss << (int64) element(idx); break;
            /* uints */
            case DataType::UINT8_T:  oss << (uint64) element(idx); break;
            case DataType::UINT16_T: oss << (uint64) element(idx); break;
            case DataType::UINT32_T: oss << (uint64) element(idx); break;
            case DataType::UINT64_T: oss << (uint64) element(idx); break;
            /* floats */
            case DataType::FLOAT32_T: oss << (float64) element(idx); break;
            case DataType::FLOAT64_T: oss << (float64) element(idx); break;
        
        }
        first=false;
    }

    if(nele > 1)
        oss << "]";
}


//============================================
template <typename T> 
void            
DataArray<T>::set(const int8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const  int16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const int32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const  int64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const  uint8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const  uint16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const uint32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <typename T> 
void            
DataArray<T>::set(const uint64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const float32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
template <typename T> 
void            
DataArray<T>::set(const float64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//============================================
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
               element_pointer(i),
               ele_bytes);
        data_ptr+=ele_bytes;
    }
}



/// Use explicit temp inst to generate the instances we need
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


};
