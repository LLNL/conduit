///
/// file: ValueArray.h
///

#include "conduit.h"

namespace conduit
{

class ValueArray
{
public:

    ValueArray(ValueType dtype, // allocates
               index_t number_of_elements,
               index_t element_width,
               index_t offset_bytes=0,
               index_t stride_bytes=0); 

    ValueArray(void *data, // wraps
               index_t number_of_elements,
               index_t element_width,
               index_t offset_bytes=0,
               index_t stride_bytes=0); 

    virtual std::string    to_schema() const;
    virtual std::string    to_string() const;

    // don't overload [], so subclasses can do so with the right return type
    
    // TODO: inline these if possible
    index_t  element_index(index_t idx)
    {
        // double check the pointer math here
        return m_offset + m_stride * idx;
    }

    void     *element(index_t idx)
    {
        // bounds check
        /*
        if(idx > m_size) return NULL;
        */
        return  m_ptr + element_index(idx);
    }
    
    void       *data() { return m_ptr;}    
    const void *data() { return m_ptr;} const 

    ValueType   dtype()               const { return m_dtype;}

    index_t     number_of_elements()  const { return m_length;}
    index_t     element_width()       const { return m_width;}
    index_t     offset()              const { return m_offset;}
    index_t     stride()              const { return m_stride;}
    index_t     total_bytes()         const;
    

    // this will be inefficent:
    uint32          as_uint32()        { return as_uint32_array()[0];}
    uint64          as_uint64()        { return as_uint64_array()[0];}
    float64         as_float64()       { return as_float64_array()[0];}
    
    // return new wrapper for each of these
    bytestr         as_bytestr();      { return ((bytestr*)this);}
    uint32_array    as_uint32_array()  { return ((uint32_array*)this);}
    uint64_array    as_uint64_array()  { return ((uint64_array*)this);}
    float64_array   as_float64_array() { return ((float64_array*)this);}


protected:
     void      *m_ptr;     // data 
     ValueType  m_dtype;   // data type (for now: {uint32,uint64,float64})
     index_t    m_offset;  // bytes from m_ptr to start of array
     index_t    m_width;   // bytes per element
     index_t    m_stride;  // bytes between start of current and start of next
     index_t    m_length;  // number of entries
     bool       m_alloced; // does this obj own the memory

    // TODO: Other things we need but may not need to touch for the hackathon:
    // Endianness m_endian;  // {big,little,default}
    // index_t    m_pad;     // unused space after data in bytes
};


class uint32_array : public ValueArray
{
public:
    uint32_array(index_t size, // allocates
                 index_t offset=0,
                 index_t stride=0);
                  
    uint32_array(uint32 *data, // wraps
                 index_t size,
                 index_t offset=0,
                 index_t stride=0);

    uint32_array(void   *data, // wraps
                 index_t size,
                 index_t offset=0,
                 index_t stride=0);
    // need full set of copy constructors, these won't own data, they will wrap
    uint32_array(uint32_array &data);


    uint32 &value(index_t idx)                                    
    {return *((uint32*)element(idx));}

    const uint32 &value(index_t idx) const
    {return *((uint32*)element(idx));}
    
    uint32 &operator[](index_t idx)
    {return value(idx);}

    const uint32 &operator[](index_t idx) const
    {return value(idx);}
};


class uint64_array : public ValueArray
{
public:
    uint64_array(index_t length, // allocates
                 index_t offset=0,
                 index_t stride=0);
                  
    uint64_array(uint32 *data, // wraps
                 index_t length,
                 index_t offset=0,
                 index_t stride=0);

    uint64_array(void   *data, // wraps
                 index_t length,
                 index_t offset=0,
                 index_t stride=0);

     // need full set of copy constructors, these won't own data, they will wrap
     uint64_array(uint64_array &data);

    uint64 &value(uint64 idx)                                    
    {return *((uint64*)element(idx));}

    const uint64 &value(uint64 idx) const;
    {return *((uint64*)element(idx));}
    
    uint64 &operator[](index_t idx)
    {return value(idx);}

    const uint64 &operator[](index_t idx) const
    {return value(idx);}

};

class float64_array : public ValueArray
{
public:
    float64_array(index_t length, // allocates
                  index_t offset=0,
                  index_t stride=0);
                  
    float64_array(uint32 *data, // wraps
                  index_t length,
                  index_t offset=0,
                  index_t stride=0);

    float64_array(void   *data, // wraps
                  index_t length,
                  index_t offset=0,
                  index_t stride=0);

    // need full set of copy constructors, these won't own data, they will wrap
    float64_array(uint64_array &data);

    float64 &value(index_t idx)                                    
    {return *((float64*)element(idx));}

    const float64 &value(index_t idx) const
    {return *((float64*)element(idx));}
    
    float64 &operator[](index_t idx)
    {return value(idx);}

    const float64 &operator[](index_t idx) const
    {return value(idx);}

};

class bytestr : public ValueArray
{
public:
    bytestr(index_t length); // allocates

    bytestr(const std::string &data); //allocates
    
    bytestr(const char *data, // wraps
            index_t length=0, // 0 implies use strlen()
            index_t offset=0);

    bytestr(void    *data, // wraps
            index_t length,
            index_t offset=0);

    uint8  *value()
    {return (uint8*)element(0);}
    
    virtual std::string to_string()
    {return std::string(value());}
    
};

};


