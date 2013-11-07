///
/// file: DataArray.h
///

#include "conduit.h"

namespace conduit
{

class DataArray
{
public:
    DataArray(const std::string &dtype_name, // allocates
              index_t size=1,
              index_t offset=0,
              index_t stride=0)

    DataArray(DataType dt, // allocates
              index_t size=1,
              index_t offset=0,
              index_t stride=0); 
    DataArray(void *data, // wraps
              index_t size,
              index_t offset=0,
              index_t stride=0)); 

    // TODO: inline these if possible
    void     *element(index_t idx)
    {
        // bounds check
        /*
        if(idx > m_size) return NULL;
        */
        // double check the pointer math here
        void *ele_ptr = m_ptr + m_offset + m_stride * idx;
        return ele_ptr;
    }

    void     *ptr() { return m_ptr;}    
    DataType  dtype()   const { return m_dtype;}
    index_t   offset()  const { return m_offset;}
    index_t   width()   const { return m_width;}
    index_t   stride()  const { return m_stride;}
    index_t   size()    const { return m_size;}    
    bool      alloced() const { return m_alloced;}  


protected:
     void      *m_ptr;     // data 
     DataType   m_dtype;   // data type (for now: {uint32,uint64,float64})
     index_t    m_offset;  // bytes from m_ptr to start of array
     index_t    m_width    // bytes per element
     index_t    m_stride;  // bytes between start of current and start of next
     index_t    m_size;    // number of entries
     bool       m_alloced; // do we own the memory

    // TODO: Other things we need but may not need to touch for the hackathon:
    // Endianness m_endian;  // {big,little,default}
    // index_t    m_pad;     // unused space after data in bytes
};


class uint32_darray : public DataArray
{
public:
    uint32_darray(index_t size=1, // allocates
                  index_t offset=0,
                  index_t stride=0);
                  
    uint32_darray(uint32 *data, // wraps
                  index_t size=1,
                  index_t offset=0,
                  index_t stride=0);

    uint32_darray(void   *data, // wraps
                  index_t size=1,
                  index_t offset=0,
                  index_t stride=0);

    uint32 &value(index_t idx)                                    
    {return *((uint32*)element(idx));}
    
    uint32 &operator[](index_t idx)
    {return value(idx);}
};


class uint64_darray : public DataArray
{
public:
    uint64_darray(index_t size=1, // allocates
                  index_t offset=0,
                  index_t stride=0);
                  
    uint64_darray(uint32 *data, // wraps
                  index_t size=1,
                  index_t offset=0,
                  index_t stride=0);

    uint64_darray(void   *data, // wraps
                  index_t size=1,
                  index_t offset=0,
                  index_t stride=0);

    uint64 &value(uint64 idx)                                    
    {return *((uint64*)element(idx));}
    
    uint64 &operator[](index_t idx)
    {return value(idx);}
};

class float64_darray : public DataArray
{
public:
    float64_darray(index_t size=1, // allocates
                   index_t offset=0,
                   index_t stride=0);
                  
    float64_darray(uint32 *data, // wraps
                   index_t size=1,
                   index_t offset=0,
                   index_t stride=0);

    float64_darray(void   *data, // wraps
                   index_t size=1,
                   index_t offset=0,
                   index_t stride=0);

    float64 &value(index_t idx)                                    
    {return *((float64*)element(idx));}
    
    float64 &operator[](index_t idx)
    {return value(idx);}
};

class String : public DataArray
{
public:
    String(index_t size); // allocates

    String(const std::string &data); //allocates
    
    String(const char *data, // wraps
           index_t offset=0);

    String(void *data, // wraps
           index_t offset=0,
           index_t size=1);

    const char *value()
    {return (const char*)element(idx);}
    
    std::string to_string()
    {return std::string(value());}
    
};

};


