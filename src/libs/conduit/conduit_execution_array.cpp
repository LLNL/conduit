// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_execution_array.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_execution_array.hpp"

//-----------------------------------------------------------------------------
// -- standard includes -- 
//-----------------------------------------------------------------------------
#include <algorithm>
#include <cstring>
#include <limits>

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "conduit_node.hpp"
#include "conduit_utils.hpp"
#include "conduit_log.hpp"
#include "conduit_memory_manager.hpp"

// Easier access to the Conduit logging functions
using namespace conduit::utils;

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
//
// -- conduit::ExecutionArray public methods --
//
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
template <typename T> 
ExecutionArray<T>::ExecutionArray()
: m_node_ptr(nullptr),
  m_other_ptr(nullptr),
  m_other_dtype(DataType::empty()),
  m_do_i_own_it(false),
  m_data(nullptr),
  m_offset(0),
  m_stride(0)
{} 

//---------------------------------------------------------------------------// 
template <typename T> 
ExecutionArray<T>::ExecutionArray(const ExecutionArray<T> &array)
: m_node_ptr(array.m_node_ptr),
  m_other_ptr(array.m_other_ptr),
  m_other_dtype(array.m_other_dtype),
  m_do_i_own_it(array.m_do_i_own_it),
  m_data(array.m_data),
  m_offset(array.m_offset),
  m_stride(array.m_stride)
{}

//---------------------------------------------------------------------------//
template <typename T> 
ExecutionArray<T>::ExecutionArray(Node &node)
: m_node_ptr(&node),
  m_other_ptr(nullptr),
  m_other_dtype(DataType::empty()),
  m_do_i_own_it(false),
  m_data(node.data_ptr()),
  m_offset(node.dtype().offset()),
  m_stride(node.dtype().stride())
{}


//---------------------------------------------------------------------------//
template <typename T> 
ExecutionArray<T>::ExecutionArray(Node *node)
: m_node_ptr(node),
  m_other_ptr(nullptr),
  m_other_dtype(DataType::empty()),
  m_do_i_own_it(false),
  m_data(node->data_ptr()),
  m_offset(node->dtype().offset()),
  m_stride(node->dtype().stride())
{}


//---------------------------------------------------------------------------//
template <typename T> 
ExecutionArray<T>::ExecutionArray(const Node *node)
: m_node_ptr(node),
  m_other_ptr(nullptr),
  m_other_dtype(DataType::empty()),
  m_do_i_own_it(false),
  m_data(node->data_ptr()),
  m_offset(node->dtype().offset()),
  m_stride(node->dtype().stride())
{}


//---------------------------------------------------------------------------//
template <typename T> 
ExecutionArray<T>::~ExecutionArray()
{
	if (m_do_i_own_it)
	{
	    if (DeviceMemory::is_device_ptr(m_other_ptr))
	    {
	        DeviceMemory::deallocate(m_other_ptr);
	    }
	    else
	    {
	        HostMemory::deallocate(m_other_ptr);
	    }
	}
}

//---------------------------------------------------------------------------//
template <typename T> 
ExecutionArray<T> &
ExecutionArray<T>::operator=(const ExecutionArray<T> &array)
{
    if(this != &array)
    {
        m_node_ptr = array.m_node_ptr;
        m_other_ptr = array.m_other_ptr;
        m_other_dtype = array.m_other_dtype;
        m_do_i_own_it = array.m_do_i_own_it;
        m_data = array.m_data;
        m_offset = array.m_offset;
        m_stride = array.m_stride;
    }
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T> 
T &
ExecutionArray<T>::element(index_t idx)
{ 
    return (*(T*)(element_ptr(idx)));
}

//---------------------------------------------------------------------------//
template <typename T> 
T &             
ExecutionArray<T>::element(index_t idx) const 
{ 
    return (*(T*)(element_ptr(idx)));
}

//---------------------------------------------------------------------------//
template <typename T> 
bool
ExecutionArray<T>::compatible(const ExecutionArray<T> &array) const 
{ 
    return dtype().compatible(array.dtype());
}

//---------------------------------------------------------------------------//
template <typename T> 
bool
ExecutionArray<T>::diff(const ExecutionArray<T> &array, Node &info, const float64 epsilon) const 
{ 
    const std::string protocol = "data_array::diff";
    bool res = false;
    info.reset();

    index_t t_nelems = number_of_elements();
    index_t o_nelems = array.number_of_elements();


    if(dtype().is_char8_str())
    {
        // since char8_str are null termed strings
        // diff should obey those semantics
        //
        // buffers may not always be 100% equal

        const char *t_data = nullptr;
        const char *o_data = nullptr;

        uint8 *t_compact_data = nullptr;
        uint8 *o_compact_data = nullptr;

        // conduit stores char8_strs with null terms
        // if the data array size is 0, we have a empty
        // case that is distinct from an empty string.
        if(t_nelems > 0)
        {
            if(dtype().is_compact())
            {
                t_data = (const char*)element_ptr(0);
            }
            else
            {
                t_compact_data = new uint8[(size_t)dtype().bytes_compact()];
                compact_elements_to(t_compact_data);
                t_data = (const char*)t_compact_data;
            }
        }

        if(o_nelems > 0)
        {
            if(array.dtype().is_compact())
            {
                o_data = (const char*)array.element_ptr(0);
            }
            else
            {
                o_compact_data = new uint8[(size_t)array.dtype().bytes_compact()];
                array.compact_elements_to(o_compact_data);
                o_data = (const char*)o_compact_data;
            }
        }

        // for debugging:
        // if(t_data)
        //     std::cout << "t_data " << t_data <<std::endl;
        // if(o_data)
        //     std::cout << "o_data " << o_data <<std::endl;

        if(t_nelems == 0 && o_nelems == 0)
        {
            // ok, both are empty buffers
        }
        // t_data is null, array is len 0
        else if( t_nelems == 0 )
        {
            std::ostringstream oss;
            oss << "data string mismatch ("
                << " [empty buffer] "
                << " vs "
                << "\"" << o_data << "\""
                << ")";
            log::error(info, protocol, oss.str());
            res = true;
        }
        // o_data is null, array is len 0
        else if( o_nelems == 0)
        {
            std::ostringstream oss;
            oss << "data string mismatch ("
                << "\"" << t_data << "\""
                << " vs "
                << " [empty buffer] "
                << ")";
            log::error(info, protocol, oss.str());
            res = true;
        }
        // all other cases use strcmp
        else if(strcmp(t_data,o_data) != 0)
        {
            std::ostringstream oss;
            oss << "data string mismatch ("
                << "\"" << t_data << "\""
                << " vs "
                << "\"" << o_data << "\""
                << ")";
            log::error(info, protocol, oss.str());
            res = true;
        }

        if(t_compact_data)
        {
            delete [] t_compact_data;
        }

        if(o_compact_data)
        {
            delete [] o_compact_data;
        }
    }
    else if(t_nelems != o_nelems)
    {
        std::ostringstream oss;
        oss << "data length mismatch ("
            << t_nelems
            << " vs "
            << o_nelems
            << ")";
        log::error(info, protocol, oss.str());
        res = true;
    }
    else
    {
        Node &info_value = info["value"];
        info_value.set(DataType(array.dtype().id(), t_nelems));
        T* info_ptr = (T*)info_value.data_ptr();

        for(index_t i = 0; i < t_nelems; i++)
        {
            info_ptr[i] = (*this)[i] - array[i];
            if(dtype().is_floating_point())
            {
                res |= info_ptr[i] > epsilon || info_ptr[i] < -epsilon;
            }
            else
            {
                res |= (*this)[i] != array[i];
            }
        }

        if(res)
        {
            log::error(info, protocol, "data item(s) mismatch; see 'value' section");
        }
    }

    log::validation(info, !res);

    return res;
}

//---------------------------------------------------------------------------//
template <typename T> 
bool
ExecutionArray<T>::diff_compatible(const ExecutionArray<T> &array, Node &info, const float64 epsilon) const 
{ 
    const std::string protocol = "data_array::diff_compatible";
    bool res = false;
    info.reset();

    index_t t_nelems = number_of_elements();
    index_t o_nelems = array.number_of_elements();

    // if we have a string, look that compat string
    // is a substring that starts at the beginning
    // of the test string
    if(dtype().is_char8_str())
    {

        const char *t_data = nullptr;
        const char *o_data = nullptr;

        uint8 *t_compact_data = nullptr;
        uint8 *o_compact_data = nullptr;

        // conduit stores char8_strs with null terms
        // if the data array size is 0, we have a empty
        // case that is distinct from an empty string.
        if(t_nelems > 0)
        {
            if(dtype().is_compact())
            {
                t_data = (const char*)element_ptr(0);
            }
            else
            {
                t_compact_data = new uint8[(size_t)dtype().bytes_compact()];
                compact_elements_to(t_compact_data);
                t_data = (const char*)t_compact_data;
            }
        }

        if(o_nelems > 0)
        {
            if(array.dtype().is_compact())
            {
                o_data = (const char*)array.element_ptr(0);
            }
            else
            {
                o_compact_data = new uint8[(size_t)array.dtype().bytes_compact()];
                array.compact_elements_to(o_compact_data);
                o_data = (const char*)o_compact_data;
            }
        }

        // for debugging:
        // if(t_data)
        //     std::cout << "t_data " << t_data <<std::endl;
        // if(o_data)
        //     std::cout << "o_data " << o_data <<std::endl;

        if(t_nelems == 0 && o_nelems == 0)
        {
            // ok, both are empty buffers
        }
        // t_data is null, array is len 0
        else if( t_nelems == 0 )
        {
            std::ostringstream oss;
            oss << "data string mismatch ("
                << " [empty buffer] "
                << " vs "
                << "\"" << o_data << "\""
                << ")";
            log::error(info, protocol, oss.str());
            res = true;
        }
        // o_data is null, array is len 0
        else if( o_nelems == 0)
        {
            std::ostringstream oss;
            oss << "data string mismatch ("
                << "\"" << t_data << "\""
                << " vs "
                << " [empty buffer] "
                << ")";
            log::error(info, protocol, oss.str());
            res = true;
        }
        // standard compat size check
        // (t_data len must be less than o_data len)
        else if(strlen(t_data) > strlen(o_data))
        {
            std::ostringstream oss;
            oss << "arg string length incompatible ("
                << t_nelems
                << " vs "
                << o_nelems
                << ")";
            log::error(info, protocol, oss.str());
            res = true;
        }
        // all other cases use strstr
        else
        {
            // check that t_data is a substr of
            // o_data, and that substr 
            // starts at the beginning of o_data
            const char *found = strstr(o_data,t_data);
            // the substr should be found at the start
            // of o_data
            if(found != o_data)
            {
                std::ostringstream oss;
                oss << "data string mismatch ("
                    << "\"" << t_data << "\""
                    << " vs "
                    << "\"" << o_data << "\""
                    << ")";
                log::error(info, protocol, oss.str());
                res = true;
            }
        }

        if(t_compact_data)
        {
            delete [] t_compact_data;
        }

        if(o_compact_data)
        {
            delete [] o_compact_data;
        }
    }
    else if(t_nelems > o_nelems)
    {
        std::ostringstream oss;
        oss << "arg data length incompatible ("
            << t_nelems
            << " vs "
            << o_nelems
            << ")";
        log::error(info, protocol, oss.str());
        res = true;
    }
    else
    {
        Node &info_value = info["value"];
        info_value.set(DataType(array.dtype().id(), t_nelems));
        T* info_ptr = (T*)info_value.data_ptr();

        for(index_t i = 0; i < t_nelems; i++)
        {
            info_ptr[i] = (*this)[i] - array[i];
            if(dtype().is_floating_point())
            {
                res |= info_ptr[i] > epsilon || info_ptr[i] < -epsilon;
            }
            else
            {
                res |= (*this)[i] != array[i];
            }
        }

        if(res)
        {
            log::error(info, protocol, "data item(s) mismatch; see diff below");
        }
    }

    log::validation(info, !res);

    return res;
}

//---------------------------------------------------------------------------// 
///
/// Summary Stats Helpers
///
//---------------------------------------------------------------------------// 

//---------------------------------------------------------------------------// 
template <typename T>
T
ExecutionArray<T>::min()  const
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
ExecutionArray<T>::max() const
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
ExecutionArray<T>::sum() const
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
ExecutionArray<T>::mean() const
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
ExecutionArray<T>::count(T val) const
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
void
ExecutionAccessor<T>::use_with(conduit::execution::policy policy)
{
    // we are being asked to execute on the device
    if (policy == conduit::execution::policy::Device)
    {
        // data is already on the device
        if (DeviceMemory::is_device_ptr(m_data))
        {
            // Do nothing
        }
        else // m_data is on the host
        {
            // if we started out on the host
            if (m_node_ptr->data_ptr() == m_data)
            {
                CONDUIT_ASSERT(m_other_ptr == nullptr,
                    "Using accessor in this way will result in a memory leak.");

                // allocate new memory and create a new dtype
                m_other_ptr = DeviceMemory::allocate(dtype().element_bytes() * number_of_elements());
                m_do_i_own_it = true;
                m_other_dtype = DataType(dtype().id(),
                                         number_of_elements(),
                                         0, // offset is 0
                                         DataType::default_bytes(dtype().id()), // stride
                                         dtype().element_bytes(),
                                         dtype().endianness());

                // copy data
                utils::conduit_memcpy_strided_elements(m_other_ptr,
                                                       number_of_elements(),
                                                       dtype().element_bytes(),
                                                       m_other_dtype.stride(),
                                                       m_data,
                                                       dtype().stride());

                // change where our data pointer points and update offset and stride
                m_data = m_other_ptr;
                m_offset = m_other_dtype.offset();
                m_stride = m_other_dtype.stride();
            }
            else // we started out on the device
            {
                CONDUIT_ASSERT(m_data == m_other_ptr, "TODO");

                // call sync to bring our copy of the data on the host back to the device
                sync();

                // dealloc the ptr on the host now that we have copied back
                HostMemory::deallocate(m_data);
                m_do_i_own_it = false;
                m_other_dtype = DataType::empty();

                // set m_data to device data and update offset and stride
                m_data = m_node_ptr->data_ptr();
                // the order of operations is important here; changing the pointer
                // will change the result of calling dtype().
                m_offset = dtype().offset();
                m_stride = dtype().stride();

                // reset m_other_ptr
                m_other_ptr = nullptr;
            }
        }
    }
    // TODO do we support other cases here? Serial, Device, Cuda, Hip, OpenMP
    else // we are being asked to execute on the host
    {
        // data is already on the host
        if (! DeviceMemory::is_device_ptr(m_data))
        {
            // Do nothing
        }
        else // m_data is on the device
        {
            // if we started out on the device
            if (m_node_ptr->data_ptr() == m_data)
            {
                CONDUIT_ASSERT(m_other_ptr == nullptr,
                    "Using accessor in this way will result in a memory leak.");

                // allocate new memory and create a new dtype
                m_other_ptr = HostMemory::allocate(dtype().element_bytes() * number_of_elements());
                m_do_i_own_it = true;
                m_other_dtype = DataType(dtype().id(),
                                         number_of_elements(),
                                         0, // offset is 0
                                         DataType::default_bytes(dtype().id()), // stride
                                         dtype().element_bytes(),
                                         dtype().endianness());

                // copy data
                utils::conduit_memcpy_strided_elements(m_other_ptr,
                                                       number_of_elements(),
                                                       dtype().element_bytes(),
                                                       m_other_dtype.stride(),
                                                       m_data,
                                                       dtype().stride());

                // change where our data pointer points and update offset and stride
                m_data = m_other_ptr;
                m_offset = m_other_dtype.offset();
                m_stride = m_other_dtype.stride();
            }
            else // we started out on the host
            {
                CONDUIT_ASSERT(m_data == m_other_ptr, "TODO");

                // call sync to bring our copy of the data on the device back to the host
                sync();

                // dealloc the ptr on the host now that we have copied back
                DeviceMemory::deallocate(m_data);
                m_do_i_own_it = false;
                m_other_dtype = DataType::empty();

                // set m_data to host data and update offset and stride
                m_data = m_node_ptr->data_ptr();
                m_offset = dtype().offset();
                m_stride = dtype().stride();

                // reset m_other_ptr
                m_other_ptr = nullptr;
            }
        }
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionAccessor<T>::sync()
{
    // if the ptrs don't point to the same place
    if (m_data != m_node_ptr->data_ptr())
    {
        if (!(m_node_ptr->dtype().compatible(dtype()) && 
              number_of_elements() == m_node_ptr->number_of_elements()))
        {
            m_node_ptr->set(dtype());
        }
        utils::conduit_memcpy_strided_elements(m_node_ptr->data_ptr(),
                                               number_of_elements(),
                                               m_node_ptr->dtype().element_bytes(),
                                               m_node_ptr->dtype().stride(),
                                               m_data,
                                               m_stride);
    }
}


//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionAccessor<T>::assume()
{
    // if the ptrs don't point to the same place
    if (m_data != m_node_ptr->data_ptr())
    {
        CONDUIT_ASSERT(m_data == m_other_ptr, "TODO");

        // reset will deallocate the data the node points to
        m_node_ptr->reset();
        m_node_ptr->schema_ptr()->set(dtype());
        m_node_ptr->set_data_ptr(m_data);

        // we no longer own the data since we have given it to node
        m_other_ptr = nullptr;
        m_do_i_own_it = false;
        m_other_dtype = DataType::empty();
    }
}

//---------------------------------------------------------------------------// 
template <typename T>
std::string 
ExecutionArray<T>::to_string(const std::string &protocol) const
{
    std::ostringstream oss;
    to_string_stream(oss,protocol);
    return oss.str();
}

//---------------------------------------------------------------------------// 
template <typename T>
void
ExecutionArray<T>::to_string_stream(std::ostream &os, 
                               const std::string &protocol) const
{
    if(protocol == "yaml")
    {
        to_yaml_stream(os);
    }
    else if(protocol == "json")
    {
        to_json_stream(os);
    }
    else
    {
        // unsupported
        CONDUIT_ERROR("Unknown DataType::to_string protocol:" << protocol
                     <<"\nSupported protocols:\n" 
                     <<" json, yaml");
    }

}

//---------------------------------------------------------------------------//
template <typename T>
std::string
ExecutionArray<T>::to_string_default() const
{ 
    return to_string();
}


//---------------------------------------------------------------------------//
template <typename T> 
std::string
ExecutionArray<T>::to_json() const 
{ 
    std::ostringstream oss;
    to_json_stream(oss);
    return oss.str(); 
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::to_json(std::ostream &os) const 
{ 
    to_json_stream(os);
}
//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::to_json_stream(std::ostream &os) const 
{ 
    index_t nele = number_of_elements();
    // note: nele == 0 case: 
    // https://github.com/LLNL/conduit/issues/992
    // we want empty arrays to display as [] not empty string
    if(nele == 0 || nele > 1)
        os << "[";

    bool first=true;
    for(index_t idx = 0; idx < nele; idx++)
    {
        if(!first)
            os << ", ";
        switch(dtype().id())
        {
            // ints 
            case DataType::INT8_ID:
            case DataType::INT16_ID: 
            case DataType::INT32_ID:
            case DataType::INT64_ID:
            {
                 os << (int64) element(idx);
                 break;
            }
            // uints
            case DataType::UINT8_ID:
            case DataType::UINT16_ID:
            case DataType::UINT32_ID:
            case DataType::UINT64_ID:
            {
                os << (uint64) element(idx);
                break;
            }
            // floats 
            case DataType::FLOAT32_ID: 
            case DataType::FLOAT64_ID: 
            {
                std::string fs = utils::float64_to_string((float64)element(idx));
                //check for inf and nan
                // looking for 'n' covers inf and nan
                bool inf_or_nan = fs.find('n') != std::string::npos;
                
                if(inf_or_nan)
                    os << "\"";
                
                os << fs;

                if(inf_or_nan)
                    os << "\"";
                break;
            }
            default:
            {
                CONDUIT_ERROR("Leaf type \"" 
                              <<  dtype().name()
                              << "\"" 
                              << "is not supported in conduit::ExecutionArray.")
            }
        }
        first=false;
    }
    // note: nele == 0 case: 
    // https://github.com/LLNL/conduit/issues/992
    // we want empty arrays to display as [] not empty string
    if(nele == 0 || nele > 1)
        os << "]";
}

//---------------------------------------------------------------------------//
template <typename T> 
std::string
ExecutionArray<T>::to_yaml() const 
{ 
    std::ostringstream oss;
    to_yaml_stream(oss);
    return oss.str(); 
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::to_yaml_stream(std::ostream &os) const 
{ 
    // yep, its the same as to_json_stream ...
    to_json_stream(os);;
}

//---------------------------------------------------------------------------//
// ExecutionArray::set() signed integers single element
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, int8 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, int16 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, int32 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, int64 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
// ExecutionArray::set() unsigned integers single element
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, uint8 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, uint16 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, uint32 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, uint64 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
// ExecutionArray::set() floating point single element
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, float32 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(index_t ele_idx, float64 value)
{ 
    this->element(ele_idx) = (T)value;
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(const int8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(const  int16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const int32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const  int64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const  uint8 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const  uint16 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const uint32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const uint64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const float32 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void            
ExecutionArray<T>::set(const float64 *values, index_t num_elements)
{ 
    for(index_t i=0;i<num_elements;i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// Set from std::initializer_list 
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<int8> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<int8>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<int16> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<int16>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<int32> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<int32>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<int64> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<int64>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<uint8> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<uint8>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<uint16> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<uint16>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<uint32> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<uint32>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<uint64> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<uint64>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<float32> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<float32>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(const std::initializer_list<float64> &values)
{
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<float64>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}


//---------------------------------------------------------------------------//
// Set from std::initializer_list  c native gap methods
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<char> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<char>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_CHAR
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<signed char> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<signed char>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<unsigned char> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<unsigned char>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_CHAR
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_SHORT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<short> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<short>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<unsigned short> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<unsigned short>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_SHORT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_INT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<int> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<int>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<unsigned int> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<unsigned int>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_INT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_LONG
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<long> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<long>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<unsigned long> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<unsigned long>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_LONG
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
#if defined(CONDUIT_HAS_LONG_LONG) && !defined(CONDUIT_USE_LONG_LONG)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<long long> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<long long>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<unsigned long long> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<unsigned long long>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_FLOAT
//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<float> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<float>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_FLOAT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_DOUBLE
//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const std::initializer_list<double> &values)
{ 
    index_t idx = 0;
    index_t num_elems = dtype().number_of_elements();
    // iterate and set up to the number of elements of this array
    std::initializer_list<double>::const_iterator itr;
    for( itr = values.begin();
         idx < num_elems && itr != values.end();
         ++itr, idx++)
    {
        this->element(idx) = (T)*itr;
    }
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_DOUBLE
//---------------------------------------------------------------------------//



//-----------------------------------------------------------------------------
// fill
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
/// signed integer fill
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::fill(int8 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
template <typename T> 
void
ExecutionArray<T>::fill(int16 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
template <typename T> 
void
ExecutionArray<T>::fill(int32 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
template <typename T> 
void
ExecutionArray<T>::fill(int64 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
/// unsigned integer fill
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::fill(uint8 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
template <typename T> 
void
ExecutionArray<T>::fill(uint16 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
template <typename T> 
void
ExecutionArray<T>::fill(uint32 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
template <typename T> 
void
ExecutionArray<T>::fill(uint64 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
/// floating point fill
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::fill(float32 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//-----------------------------------------------------------------------------
template <typename T> 
void
ExecutionArray<T>::fill(float64 value)
{ 
    for(index_t i=0;i < dtype().number_of_elements(); i++)
    {
        this->element(i) = (T)value;
    }
}

//---------------------------------------------------------------------------//
// assign operator overloads for initializer_list
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<int8> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<int16> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<int32> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<int64> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<uint8> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<uint16> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<uint32> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<uint64> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<float32> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<float64> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
// Set from std::initializer_list  c native gap methods
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<char> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_CHAR
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<signed char> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<unsigned char> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_CHAR
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_SHORT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<short> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<unsigned short> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_SHORT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_INT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<int> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<unsigned int> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_INT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_LONG
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<long> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<unsigned long> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_LONG
//---------------------------------------------------------------------------//


//---------------------------------------------------------------------------//
#if defined(CONDUIT_HAS_LONG_LONG) && !defined(CONDUIT_USE_LONG_LONG)
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<long long> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<unsigned long long> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_LONG_LONG
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_FLOAT
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<float> &values)
{
    set(values);
    return *this;
}


//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_FLOAT
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
#ifndef CONDUIT_USE_DOUBLE
//---------------------------------------------------------------------------//
//---------------------------------------------------------------------------//
template <typename T>
ExecutionArray<T> &
ExecutionArray<T>::operator=(const std::initializer_list<double> &values)
{
    set(values);
    return *this;
}

//---------------------------------------------------------------------------//
#endif // CONDUIT_USE_DOUBLE
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
//***************************************************************************//
// Set from ExecutionArray
//***************************************************************************//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Set from ExecutionArray signed integers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<int8> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<int16> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<int32> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<int64> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// Set from ExecutionArray unsigned integers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<uint8> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<uint16> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<uint32> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<uint64> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// Set from ExecutionArray floating point
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<float32> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const ExecutionArray<float64> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
//***************************************************************************//
// Set from DataAccessor
//***************************************************************************//
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Set from DataAccessor signed integers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const DataAccessor<int8> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void            
ExecutionArray<T>::set(const DataAccessor<int16> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void            
ExecutionArray<T>::set(const DataAccessor<int32> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void            
ExecutionArray<T>::set(const DataAccessor<int64> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// Set from DataAccessor unsigned integers
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const DataAccessor<uint8> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const DataAccessor<uint16> &values)
{ 
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const DataAccessor<uint32> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T>
void
ExecutionArray<T>::set(const DataAccessor<uint64> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
// Set from DataAccessor floating point
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(const DataAccessor<float32> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::set(const DataAccessor<float64> &values)
{
    index_t num_elems = dtype().number_of_elements();
    for(index_t i=0; i <num_elems; i++)
    {
        this->element(i) = (T)values[i];
    }
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::compact_elements_to(uint8 *data) const
{
    // copy all elements 
    index_t ele_bytes = DataType::default_bytes(dtype().id());

    utils::conduit_memcpy_strided_elements(data,   // dest
                    (size_t)dtype().number_of_elements(), // num eles to copy
                    ele_bytes,      // bytes per element
                    ele_bytes,      // dest stride per ele
                    element_ptr(0),             // src
                    (size_t)dtype().stride());  // src stride per ele
}

//---------------------------------------------------------------------------//
template <typename T> 
std::string
ExecutionArray<T>::to_summary_string_default() const
{ 
    return to_summary_string();
}

//---------------------------------------------------------------------------//
template <typename T> 
std::string
ExecutionArray<T>::to_summary_string(index_t threshold) const
{ 
    std::ostringstream oss;
    to_summary_string_stream(oss, threshold);
    return oss.str();
}

//---------------------------------------------------------------------------//
template <typename T> 
void
ExecutionArray<T>::to_summary_string_stream(std::ostream &os,
                                       index_t threshold) const
{ 
    // if we are less than or equal to threshold, we use to_yaml
    index_t nele = number_of_elements();

    if(nele <= threshold)
    {
        to_yaml_stream(os);
    }
    else
    {
        // if above threshold only show threshold # of values
        index_t half = threshold / 2;
        index_t bottom = half;
        index_t top = half;

        //
        // if odd, show 1/2 +1 first
        //

        if( (threshold % 2) > 0)
        {
            bottom++;
        }

        // note: nele == 0 case:
        // https://github.com/LLNL/conduit/issues/992
        // we want empty arrays to display as [] not empty string
        if(nele == 0 || nele > 1)
            os << "[";

        bool done  = (nele == 0);
        index_t idx = 0;

        while(!done)
        {
            // if not first, add a comma prefix
            if(idx > 0 )
                os << ", ";

            switch(dtype().id())
            {
                // ints
                case DataType::INT8_ID:
                case DataType::INT16_ID:
                case DataType::INT32_ID:
                case DataType::INT64_ID:
                {
                     os << (int64) element(idx);
                     break;
                }
                // uints
                case DataType::UINT8_ID:
                case DataType::UINT16_ID:
                case DataType::UINT32_ID:
                case DataType::UINT64_ID:
                {
                    os << (uint64) element(idx);
                    break;
                }
                // floats
                case DataType::FLOAT32_ID:
                case DataType::FLOAT64_ID:
                {
                    std::string fs = utils::float64_to_string((float64)element(idx));
                    //check for inf and nan
                    // looking for 'n' covers inf and nan
                    bool inf_or_nan = fs.find('n') != std::string::npos;

                    if(inf_or_nan)
                        os << "\"";

                    os << fs;

                    if(inf_or_nan)
                        os << "\"";
                    break;
                }
                default:
                {
                    CONDUIT_ERROR("Leaf type \""
                                  <<  dtype().name()
                                  << "\""
                                  << "is not supported in conduit::ExecutionArray.")
                }
            }

            idx++;

            if(idx == bottom)
            {
                idx = nele - top;
                os << ", ...";
            }

            if(idx == nele)
            {
                done = true;
            }
        }

        // note: nele == 0 case:
        // https://github.com/LLNL/conduit/issues/992
        // we want empty arrays to display as [] not empty string
        if(nele == 0 || nele > 1)
            os << "]";
    }
}



//-----------------------------------------------------------------------------
//
// -- conduit::ExecutionArray explicit instantiations for supported array types --
//
//-----------------------------------------------------------------------------
template class ExecutionArray<int8>;
template class ExecutionArray<int16>;
template class ExecutionArray<int32>;
template class ExecutionArray<int64>;

template class ExecutionArray<uint8>;
template class ExecutionArray<uint16>;
template class ExecutionArray<uint32>;
template class ExecutionArray<uint64>;

template class ExecutionArray<float32>;
template class ExecutionArray<float64>;

// gap template instantiations for c-native types

// we never use 'char' directly as a type,
// so we always need to inst the char case
template class ExecutionArray<char>;

#ifndef CONDUIT_USE_CHAR
template class ExecutionArray<signed char>;
template class ExecutionArray<unsigned char>;
#endif

#ifndef CONDUIT_USE_SHORT
template class ExecutionArray<signed short>;
template class ExecutionArray<unsigned short>;
#endif

#ifndef CONDUIT_USE_INT
template class ExecutionArray<signed int>;
template class ExecutionArray<unsigned int>;
#endif

#ifndef CONDUIT_USE_LONG
template class ExecutionArray<signed long>;
template class ExecutionArray<unsigned long>;
#endif

#if defined(CONDUIT_HAS_LONG_LONG) && !defined(CONDUIT_USE_LONG_LONG)
template class ExecutionArray<signed long long>;
template class ExecutionArray<unsigned long long>;
#endif

#ifndef CONDUIT_USE_FLOAT
template class ExecutionArray<float>;
#endif

#ifndef CONDUIT_USE_DOUBLE
template class ExecutionArray<double>;
#endif

#ifdef CONDUIT_USE_LONG_DOUBLE
    ltemplate class ExecutionArray<long double>;
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

