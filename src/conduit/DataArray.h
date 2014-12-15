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

template <typename T> 
class CONDUIT_API DataArray
{
public: 
                    DataArray(void *data, const DataType &dtype);
                    DataArray(const void *data, const DataType &dtype);
                    DataArray(const DataArray<T> &array);
                   ~DataArray();
    DataArray<T>   &operator=(const DataArray<T> &array);

    T              &operator[](index_t idx) {return element(idx);}
    T              &operator[](index_t idx) const {return element(idx);}
    
    T              &element(index_t idx);
    T              &element(index_t idx) const;

    const DataType &dtype()    const { return m_dtype;} 
    void           *data_pointer() const { return m_data;}

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

    void            compact_elements_to(uint8 *data) const;

private:
    void           *element_pointer(index_t idx)
                     {return static_cast<char*>(m_data) + m_dtype.element_index(idx);};
    const void     *element_pointer(index_t idx) const 
                     {return static_cast<char*>(m_data) + m_dtype.element_index(idx);};

    void           *m_data;
    DataType        m_dtype;
    
};


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
