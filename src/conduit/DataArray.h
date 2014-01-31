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
                    DataArray(const DataArray<T> &array);
    virtual        ~DataArray();
    DataArray<T>   &operator=(const DataArray<T> &array);

    T              &operator[](index_t idx);
    T              &operator[](index_t idx) const;

    const DataType &dtype()    const { return m_dtype;} 
    void           *data_ptr() const { return m_data;}

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

///============================================
template <class T> 
DataArray<T>::DataArray(void *data,const DataType &dtype)
: m_data(data),
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
DataArray<T>::operator[](index_t idx)
{ 
    //    std::cout << "[" << idx << "] = idx "
    //              << m_dtype.element_index(idx) << std::endl;
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}

///============================================
template <class T> 
T &             
DataArray<T>::operator[](index_t idx) const 
{ 
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}


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
