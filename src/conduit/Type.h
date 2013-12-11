///
/// file: Type.h
///

#ifndef __CONDUIT_TYPE_H
#define __CONDUIT_TYPE_H

#include "Core.h"
#include "Endianness.h"

#include <vector>
#include <string>
#include <sstream>

namespace conduit
{

class Node;
///============================================
/// DataType
///============================================
class DataType
{
public:
   static DataType empty_dtype;
   static DataType uint32_dtype;
   static DataType float64_dtype;
   
   static DataType node_dtype;
   static DataType list_dtype;
   
    typedef enum
    {
        EMPTY_T = 0, // default
        NODE_T,      // node
        LIST_T,      // list
        UINT32_T,    // uint32 and uint32_array
        UINT64_T,    // uint64 and uint64_array
        FLOAT64_T,   // float64 and float64_array
        BYTESTR_T,   // bytestr (incore c-string)
    } TypeEnum;

             DataType();
             explicit DataType(index_t id);

             DataType(const DataType& type);

             DataType(const std::string &dtype_name,
                      index_t num_elements,
                      index_t offset,
                      index_t stride,
                      index_t element_bytes,
                      index_t endianness);

             DataType(index_t dtype_id,
                      index_t num_elements,
                      index_t offset,
                      index_t stride,
                      index_t element_bytes,
                      index_t endianness);

    virtual ~DataType();

    void       reset(const DataType& type);
    void       reset(index_t dtype_id);
    void       reset(index_t dtype_id,
                     index_t num_elements,
                     index_t offset,
                     index_t stride,
                     index_t element_bytes,
                     index_t endianness);

    static index_t          name_to_id(const std::string &name);
    static std::string      id_to_name(index_t dtype);
    
    //static index_t          size_of_dtype(const std::string &name);
    //static index_t          size_of_dtype(index_t dtype);

    static DataType const  &default_dtype(index_t dtype_id);
    static DataType const  &default_dtype(const std::string &name);
       
    std::string         schema() const;
    void                schema(std::ostringstream &oss) const;
    
    index_t             id()    const { return m_id;}    
    index_t     total_bytes()   const;
    index_t     total_bytes_compact() const;
    bool        compatible_storage(const DataType& type) const;

    index_t    number_of_elements()  const { return m_num_ele;}
    index_t    offset()              const { return m_offset;}
    index_t    stride()              const { return m_stride;}
    index_t    element_bytes()       const { return m_ele_bytes;}
    index_t    endianness()          const { return m_endianness;}
    index_t    element_index(index_t idx) const;

    template<typename T>
    struct Traits { };

private:

    index_t   m_id;         // for dtype enum value
    index_t   m_num_ele;    // number of entries
    index_t   m_offset;     // bytes to start of array
    index_t   m_stride;     // bytes between start of current and start of next
    index_t   m_ele_bytes;  // bytes per element
    index_t   m_endianness;  // endianness of elements

};



template<>
struct DataType::Traits<conduit::uint32>
{
   static const DataType::TypeEnum data_type = UINT32_T;
};

template<>
struct DataType::Traits<conduit::float64>
{
   static const DataType::TypeEnum data_type = FLOAT64_T;
};



};


#endif
