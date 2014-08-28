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
/// file: DataArray.h
///

#ifndef __CONDUIT_DATA_ARRAY_H
#define __CONDUIT_DATA_ARRAY_H

#include "Core.h"
#include "DataType.h"

namespace conduit
{

template <class T> 
class DataArray
{
public: 
                    DataArray(void *data, const DataType &dtype);
                    DataArray(const void *data, const DataType &dtype);
                    DataArray(const DataArray<T> &array);
    virtual        ~DataArray();
    DataArray<T>   &operator=(const DataArray<T> &array);

    T              &operator[](index_t idx) {return element(idx);}
    T              &operator[](index_t idx) const {return element(idx);}
    
    T              &element(index_t idx);
    T              &element(index_t idx) const;

    const DataType &dtype()    const { return m_dtype;} 
    void           *data_ptr() const { return m_data;}

    index_t         number_of_elements() const {return m_dtype.number_of_elements();}

    std::string     to_json() const;
    void            to_json(std::ostringstream &oss) const;

    void            set(const bool8 *values, index_t num_elements);

    void            set(const int8  *values, index_t num_elements);
    void            set(const int16 *values, index_t num_elements);
    void            set(const int32 *values, index_t num_elements);
    void            set(const int64 *values, index_t num_elements);

    void            set(const uint8   *values, index_t num_elements);
    void            set(const uint16  *values, index_t num_elements);
    void            set(const uint32  *values, index_t num_elements);
    void            set(const uint64  *values, index_t num_elements);
    
    void            set(const float32 *values, index_t num_elements);
    void            set(const float64 *values, index_t num_elements);


    void            set(const std::vector<bool8>   &values)  
                        {set(&values[0],values.size());}
    
    void            set(const std::vector<int8>    &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int16>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int32>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int64>   &values)
                        {set(&values[0],values.size());}

    void            set(const std::vector<uint8>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint16>  &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint32>  &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint64>  &values)
                        {set(&values[0],values.size());}
    
    void            set(const std::vector<float32> &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<float64> &values)
                        {set(&values[0],values.size());}

private:
    void           *element_pointer(index_t idx)
                     {return static_cast<char*>(m_data) + m_dtype.element_index(idx);};
    const void     *element_pointer(index_t idx) const 
                     {return static_cast<char*>(m_data) + m_dtype.element_index(idx);};

    void           *m_data;
    DataType        m_dtype;
    
};

///============================================
/// DataArray
///============================================

/// TODO: Move these to DataArray.cpp, after we figure out the right exp temp inst calls. 

///============================================
template <class T> 
DataArray<T>::DataArray(void *data,const DataType &dtype)
: m_data(data),
  m_dtype(dtype)
{}

///============================================
template <class T> 
DataArray<T>::DataArray(const void *data,const DataType &dtype)
: m_data(const_cast<void*>(data)),
  m_dtype(dtype)
{}


///============================================ 
template <class T> 
DataArray<T>::DataArray(const DataArray<T> &array)
: m_data(array.m_data),
  m_dtype(array.m_dtype)
{}

///============================================
template <class T> 
DataArray<T>::~DataArray()
{} // all data is external

///============================================
template <class T> 
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

///============================================
template <class T> 
T &
DataArray<T>::element(index_t idx)
{ 
    //    std::cout << "[" << idx << "] = idx "
    //              << m_dtype.element_index(idx) << std::endl;
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}

///============================================
template <class T> 
T &             
DataArray<T>::element(index_t idx) const 
{ 
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}

///============================================
template <class T> 
std::string             
DataArray<T>::to_json() const 
{ 
    std::ostringstream oss;
    to_json(oss);
    return oss.str(); 
}

///============================================
template <class T> 
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
            /* bool */
            case DataType::BOOL8_T:
            {
                if(element(idx))
                    oss << "true";
                else
                    oss << "false";
                break;
            } 
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

template <class T> 
void            
DataArray<T>::set(const bool8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const int8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const  int16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const int32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const  int64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const  uint8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const  uint16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const uint32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const uint64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const float32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

template <class T> 
void            
DataArray<T>::set(const float64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

typedef DataArray<bool8>    bool8_array;
typedef DataArray<int8>     int8_array;
typedef DataArray<int16>    int16_array;
typedef DataArray<int32>    int32_array;
typedef DataArray<int64>    int64_array;

typedef DataArray<uint8>    uint8_array;
typedef DataArray<uint16>   uint16_array;
typedef DataArray<uint32>   uint32_array;
typedef DataArray<uint64>   uint64_array;

typedef DataArray<float32>  float32_array;
typedef DataArray<float64>  float64_array;


}

#endif
