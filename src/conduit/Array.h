///
/// file: Array.h
///

#ifndef __CONDUIT_ARRAY_H
#define __CONDUIT_ARRAY_H

#include "Core.h"
#include "Type.h"

namespace conduit
{

template <class T> 
class Array
{
public: 
                   Array(void *data, const DataType &dtype);
                   Array(const Array<T> &array);
    virtual       ~Array();
    Array<T>       &operator=(const Array<T> &array);

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
/// Array
///============================================

///============================================
template <class T> 
Array<T>::Array(void *data,const DataType &dtype)
: m_data(data),
  m_dtype(dtype)
{}

///============================================ 
template <class T> 
Array<T>::Array(const Array<T> &array)
: m_data(array.m_data),
  m_dtype(array.m_dtype)
{}

///============================================
template <class T> 
Array<T>::~Array()
{} // all data is external

///============================================
template <class T> 
Array<T> &
Array<T>::operator=(const Array<T> &array)
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
Array<T>::operator[](index_t idx)
{ 
    //    std::cout << "[" << idx << "] = idx "
    //              << m_dtype.element_index(idx) << std::endl;
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}

///============================================
template <class T> 
T &             
Array<T>::operator[](index_t idx) const 
{ 
    // TODO: endian logic
    return (*(T*)(element_pointer(idx)));
}


typedef Array<int8>     int8_array;
typedef Array<int16>    int16_array;
typedef Array<int32>    int32_array;
typedef Array<int64>    int64_array;

typedef Array<uint8>    uint8_array;
typedef Array<uint16>   uint16_array;
typedef Array<uint32>   uint32_array;
typedef Array<uint64>   uint64_array;

typedef Array<float32>  float32_array;
typedef Array<float64>  float64_array;


}

#endif
