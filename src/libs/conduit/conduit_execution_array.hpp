// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_execution_array.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_EXECUTION_ARRAY_HPP
#define CONDUIT_EXECUTION_ARRAY_HPP

#include <initializer_list>

//-----------------------------------------------------------------------------
// -- conduit  includes -- 
//-----------------------------------------------------------------------------
#include "conduit_core.hpp"
#include "conduit_data_type.hpp"
#include "conduit_utils.hpp"
#include "conduit_execution.hpp"
#include "conduit_data_accessor.hpp"
#include "conduit_execution_accessor.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- forward declarations required for conduit::ExecutionAccessor --
//-----------------------------------------------------------------------------
class Node;
template <typename T>
class DataArray;

//-----------------------------------------------------------------------------
// -- begin conduit::ExecutionArray --
//-----------------------------------------------------------------------------
///
/// class: conduit::ExecutionArray
///
/// description:
///  TODO
///
//-----------------------------------------------------------------------------
template <typename T> 
class CONDUIT_API ExecutionArray
{
public: 
    
//-----------------------------------------------------------------------------
//
// -- conduit::ExecutionArray public methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Construction and Destruction
//-----------------------------------------------------------------------------
        /// default constructor
        ExecutionArray();
        /// copy constructor
        ExecutionArray(const ExecutionArray<T> &array);
       /// Access a pointer to node data according to node dtype description.
       ExecutionArray(Node &node);
       // /// Access a const pointer to node data according to node dtype description.
       // ExecutionArray(const Node &node);
       /// Access a pointer to node data according to node dtype description.
       ExecutionArray(Node *node);
       /// Access a const pointer to node data according to node dtype description.
       ExecutionArray(const Node *node);
       /// Destructor.
       ~ExecutionArray();

    /// Assignment operator
    ExecutionArray<T>   &operator=(const ExecutionArray<T> &array);

//-----------------------------------------------------------------------------
// Data and Info Access
//-----------------------------------------------------------------------------
    typedef T ElementType;

    T              &operator[](index_t idx)
                    {return element(idx);}
    T              &operator[](index_t idx) const
                    {return element(idx);}
    
    T              &element(index_t idx);
    T              &element(index_t idx) const;

    void           *element_ptr(index_t idx)
                    {
                        return static_cast<char*>(m_data) +
                            dtype().element_index(idx);
                    };

    const void     *element_ptr(index_t idx) const 
                    {
                         return static_cast<char*>(m_data) +
                            dtype().element_index(idx);
                    };

    index_t         number_of_elements() const 
                        {return dtype().number_of_elements();}

    const DataType &dtype() const;

    const DataType &orig_dtype() const;

    const DataType &other_dtype() const;

    void           *data_ptr() const 
                        { return m_data;}

    bool            compatible(const ExecutionArray<T> &array) const;
    bool            diff(const ExecutionArray<T> &array,
                         Node &info,
                         const float64 epsilon = CONDUIT_EPSILON) const;
    bool            diff_compatible(const ExecutionArray<T> &array,
                                    Node &info,
                                    const float64 epsilon = CONDUIT_EPSILON) const;

    ///
    /// Summary Stats Helpers
    ///
    T               min()  const;
    T               max()  const;
    T               sum()  const;
    float64         mean() const;
    
    /// counts number of occurrences of given value
    index_t         count(T value) const;

//-----------------------------------------------------------------------------
// Cool Stuff
//-----------------------------------------------------------------------------
    void            use_with(conduit::execution::policy policy);

    void            sync();

    void            assume();

//-----------------------------------------------------------------------------
// Setters
//-----------------------------------------------------------------------------
    /// signed integer single element
    void            set(index_t elem_idx, int8  value);
    void            set(index_t elem_idx, int16 value);
    void            set(index_t elem_idx, int32 value);
    void            set(index_t elem_idx, int64 value);

    // unsigned integer single element
    void            set(index_t elem_idx, uint8  value);
    void            set(index_t elem_idx, uint16 value);
    void            set(index_t elem_idx, uint32 value);
    void            set(index_t elem_idx, uint64 value);

    /// floating point single element
    void            set(index_t elem_idx, float32 value);
    void            set(index_t elem_idx, float64 value);

    /// signed integer arrays
    void            set(const int8  *values, index_t num_elements);
    void            set(const int16 *values, index_t num_elements);
    void            set(const int32 *values, index_t num_elements);
    void            set(const int64 *values, index_t num_elements);

    /// unsigned integer arrays
    void            set(const uint8   *values, index_t num_elements);
    void            set(const uint16  *values, index_t num_elements);
    void            set(const uint32  *values, index_t num_elements);
    void            set(const uint64  *values, index_t num_elements);
    
    /// floating point arrays
    void            set(const float32 *values, index_t num_elements);
    void            set(const float64 *values, index_t num_elements);
    
    /// signed integer arrays via std::vector
    void            set(const std::vector<int8>    &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int16>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int32>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<int64>   &values)
                        {set(&values[0],values.size());}

    /// unsigned integer arrays via std::vector
    void            set(const std::vector<uint8>   &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint16>  &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint32>  &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<uint64>  &values)
                        {set(&values[0],values.size());}
    
    /// floating point arrays via std::vector
    void            set(const std::vector<float32> &values)
                        {set(&values[0],values.size());}
    void            set(const std::vector<float64> &values)
                        {set(&values[0],values.size());}

    //-------------------------------------------------------------------------
    // -- set for std::initializer_list types ---
    //-------------------------------------------------------------------------

    /// signed integer arrays via std::initializer_list
    void            set(const std::initializer_list<int8>    &values);
    void            set(const std::initializer_list<int16>   &values);
    void            set(const std::initializer_list<int32>   &values);
    void            set(const std::initializer_list<int64>   &values);

    /// unsigned integer arrays via std::initializer_list
    void            set(const std::initializer_list<uint8>   &values);
    void            set(const std::initializer_list<uint16>  &values);
    void            set(const std::initializer_list<uint32>  &values);
    void            set(const std::initializer_list<uint64>  &values);
    
    /// floating point arrays via std::initializer_list
    void            set(const std::initializer_list<float32> &values);
    void            set(const std::initializer_list<float64> &values);

    //-------------------------------------------------------------------------
    // --  assignment c-native gap operators for initializer_list types ---
    //-------------------------------------------------------------------------

    void set(const std::initializer_list<char> &values);

    #ifndef CONDUIT_USE_CHAR
        void set(const std::initializer_list<signed char> &values);
        void set(const std::initializer_list<unsigned char> &values);
    #endif

    #ifndef CONDUIT_USE_SHORT
        void set(const std::initializer_list<short> &values);
        void set(const std::initializer_list<unsigned short> &values);
    #endif

    #ifndef CONDUIT_USE_INT
       void set(const std::initializer_list<int> &values);
       void set(const std::initializer_list<unsigned int> &values); 
    #endif

    #ifndef CONDUIT_USE_LONG
       void set(const std::initializer_list<long> &values);
       void set(const std::initializer_list<unsigned long> &values); 
    #endif

    #if defined(CONDUIT_HAS_LONG_LONG) && !defined(CONDUIT_USE_LONG_LONG)
       void set(const std::initializer_list<long long> &values);
       void set(const std::initializer_list<unsigned long long> &values); 
    #endif

    #ifndef CONDUIT_USE_FLOAT
       void set(const std::initializer_list<float> &values);
    #endif

    #ifndef CONDUIT_USE_DOUBLE
       void set(const std::initializer_list<double> &values);
    #endif

    //-------------------------------------------------------------------------
    // -- assignment operators for std::initializer_list types ---
    //-------------------------------------------------------------------------
    // signed integer array types via std::initializer_list
    ExecutionArray &operator=(const std::initializer_list<int8>   &values);
    ExecutionArray &operator=(const std::initializer_list<int16>  &values);
    ExecutionArray &operator=(const std::initializer_list<int32>  &values);
    ExecutionArray &operator=(const std::initializer_list<int64>  &values);

    // unsigned integer array types via std::initialize_list
    ExecutionArray &operator=(const std::initializer_list<uint8>   &values);
    ExecutionArray &operator=(const std::initializer_list<uint16>  &values);
    ExecutionArray &operator=(const std::initializer_list<uint32>  &values);
    ExecutionArray &operator=(const std::initializer_list<uint64>  &values);

    // floating point array types via std::initializer_list
    ExecutionArray &operator=(const std::initializer_list<float32> &values);
    ExecutionArray &operator=(const std::initializer_list<float64> &values);

    //-------------------------------------------------------------------------
    // --  assignment c-native gap operators for initializer_list types ---
    //-------------------------------------------------------------------------

    ExecutionArray &operator=(const std::initializer_list<char> &values);

    #ifndef CONDUIT_USE_CHAR
        ExecutionArray &operator=(const std::initializer_list<signed char> &values);
        ExecutionArray &operator=(const std::initializer_list<unsigned char> &values);
    #endif

    #ifndef CONDUIT_USE_SHORT
        ExecutionArray &operator=(const std::initializer_list<short> &values);
        ExecutionArray &operator=(const std::initializer_list<unsigned short> &values);
    #endif

    #ifndef CONDUIT_USE_INT
        ExecutionArray &operator=(const std::initializer_list<int> &values);
        ExecutionArray &operator=(const std::initializer_list<unsigned int> &values);
    #endif

    #ifndef CONDUIT_USE_LONG
        ExecutionArray &operator=(const std::initializer_list<long> &values);
        ExecutionArray &operator=(const std::initializer_list<unsigned long> &values);
    #endif

    #if defined(CONDUIT_HAS_LONG_LONG) && !defined(CONDUIT_USE_LONG_LONG)
        ExecutionArray &operator=(const std::initializer_list<long long> &values);
        ExecutionArray &operator=(const std::initializer_list<unsigned long long> &values);
    #endif

    #ifndef CONDUIT_USE_FLOAT
        ExecutionArray &operator=(const std::initializer_list<float> &values);
    #endif

    #ifndef CONDUIT_USE_DOUBLE
        ExecutionArray &operator=(const std::initializer_list<double> &values);
    #endif

    /// signed integer arrays via DataArray
    void            set(const DataArray<int8>    &values);
    void            set(const DataArray<int16>   &values);
    void            set(const DataArray<int32>   &values);
    void            set(const DataArray<int64>   &values);

    /// unsigned integer arrays via DataArray
    void            set(const DataArray<uint8>   &values);
    void            set(const DataArray<uint16>  &values);
    void            set(const DataArray<uint32>  &values);
    void            set(const DataArray<uint64>  &values);
    
    /// floating point arrays via DataArray
    void            set(const DataArray<float32>  &values);
    void            set(const DataArray<float64>  &values);

    /// signed integer arrays via DataAccessor
    void            set(const DataAccessor<int8>    &values);
    void            set(const DataAccessor<int16>   &values);
    void            set(const DataAccessor<int32>   &values);
    void            set(const DataAccessor<int64>   &values);

    /// unsigned integer arrays via DataAccessor
    void            set(const DataAccessor<uint8>   &values);
    void            set(const DataAccessor<uint16>  &values);
    void            set(const DataAccessor<uint32>  &values);
    void            set(const DataAccessor<uint64>  &values);
    
    /// floating point arrays via DataAccessor
    void            set(const DataAccessor<float32>  &values);
    void            set(const DataAccessor<float64>  &values);

    /// signed integer arrays via ExecutionAccessor
    void            set(const ExecutionAccessor<int8>    &values);
    void            set(const ExecutionAccessor<int16>   &values);
    void            set(const ExecutionAccessor<int32>   &values);
    void            set(const ExecutionAccessor<int64>   &values);

    /// unsigned integer arrays via ExecutionAccessor
    void            set(const ExecutionAccessor<uint8>   &values);
    void            set(const ExecutionAccessor<uint16>  &values);
    void            set(const ExecutionAccessor<uint32>  &values);
    void            set(const ExecutionAccessor<uint64>  &values);
    
    /// floating point arrays via ExecutionAccessor
    void            set(const ExecutionAccessor<float32>  &values);
    void            set(const ExecutionAccessor<float64>  &values);

    /// signed integer arrays via ExecutionArray
    void            set(const ExecutionArray<int8>    &values);
    void            set(const ExecutionArray<int16>   &values);
    void            set(const ExecutionArray<int32>   &values);
    void            set(const ExecutionArray<int64>   &values);

    /// unsigned integer arrays via ExecutionArray
    void            set(const ExecutionArray<uint8>   &values);
    void            set(const ExecutionArray<uint16>  &values);
    void            set(const ExecutionArray<uint32>  &values);
    void            set(const ExecutionArray<uint64>  &values);
    
    /// floating point arrays via ExecutionArray
    void            set(const ExecutionArray<float32>  &values);
    void            set(const ExecutionArray<float64>  &values);

//-----------------------------------------------------------------------------
// fill
//-----------------------------------------------------------------------------
    /// signed integer fill
    void            fill(int8  value);
    void            fill(int16 value);
    void            fill(int32 value);
    void            fill(int64 value);

    /// unsigned integer fill
    void            fill(uint8  value);
    void            fill(uint16 value);
    void            fill(uint32 value);
    void            fill(uint64 value);

    /// floating point fill
    void            fill(float32 value);
    void            fill(float64 value);

//-----------------------------------------------------------------------------
// Transforms
//-----------------------------------------------------------------------------
    std::string     to_string(const std::string &protocol="json") const;
    void            to_string_stream(std::ostream &os, 
                                     const std::string &protocol="json") const;

    // NOTE(cyrush): The primary reason this function exists is to enable 
    // easier compatibility with debugging tools (e.g. totalview, gdb) that
    // have difficulty allocating default string parameters.
    std::string     to_string_default() const;

    std::string     to_json() const;
    void            to_json_stream(std::ostream &os) const;
    
    /// DEPRECATED: to_json(std::ostream &os) is deprecated in favor of 
    ///             to_json_stream(std::ostream &os)
    void            to_json(std::ostream &os) const;

    std::string     to_yaml() const;
    void            to_yaml_stream(std::ostream &os) const;

    void            compact_elements_to(uint8 *data) const;

    /// Creates a string repression for printing that limits
    /// the number of elements shown to a max number
    std::string     to_summary_string_default() const;
    std::string     to_summary_string(index_t threshold=5) const;
    void            to_summary_string_stream(std::ostream &os,
                                             index_t threshold=5) const;

//-----------------------------------------------------------------------------
// -- stdout print methods ---
//-----------------------------------------------------------------------------
    /// print a simplified json representation of the this node to std out
    void            print() const
                      {std::cout << to_summary_string() << std::endl;}


private:

//-----------------------------------------------------------------------------
//
// -- conduit::ExecutionArray private data members --
//
//-----------------------------------------------------------------------------
    Node           *m_node_ptr;

    /// holds data
    void           *m_other_ptr;
    /// holds data description
    DataType        m_other_dtype;
    
    bool            m_do_i_own_it;

    void           *m_data;
    index_t         m_offset;
    index_t         m_stride;
    
};
//-----------------------------------------------------------------------------
// -- end conduit::ExecutionArray --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- conduit::ExecutionArray typedefs for supported array types --
//
//-----------------------------------------------------------------------------

/// Note: these are also the types we explicitly instantiate.

/// signed integer arrays
typedef ExecutionArray<int8>     int8_exec_array;
typedef ExecutionArray<int16>    int16_exec_array;
typedef ExecutionArray<int32>    int32_exec_array;
typedef ExecutionArray<int64>    int64_exec_array;

/// unsigned integer arrays
typedef ExecutionArray<uint8>    uint8_exec_array;
typedef ExecutionArray<uint16>   uint16_exec_array;
typedef ExecutionArray<uint32>   uint32_exec_array;
typedef ExecutionArray<uint64>   uint64_exec_array;

/// floating point arrays
typedef ExecutionArray<float32>  float32_exec_array;
typedef ExecutionArray<float64>  float64_exec_array;

/// index type arrays
typedef ExecutionArray<index_t>  index_t_exec_array;

/// native c types arrays
typedef ExecutionArray<char>       char_exec_array;
typedef ExecutionArray<short>      short_exec_array;
typedef ExecutionArray<int>        int_exec_array;
typedef ExecutionArray<long>       long_exec_array;
#ifdef CONDUIT_HAS_LONG_LONG
typedef ExecutionArray<long long>  long_long_exec_array;
#endif


/// signed integer arrays
typedef ExecutionArray<signed char>       signed_char_exec_array;
typedef ExecutionArray<signed short>      signed_short_exec_array;
typedef ExecutionArray<signed int>        signed_int_exec_array;
typedef ExecutionArray<signed long>       signed_long_exec_array;
#ifdef CONDUIT_HAS_LONG_LONG
typedef ExecutionArray<signed long long>  signed_long_long_exec_array;
#endif


/// unsigned integer arrays
typedef ExecutionArray<unsigned char>   unsigned_char_exec_array;
typedef ExecutionArray<unsigned short>  unsigned_short_exec_array;
typedef ExecutionArray<unsigned int>    unsigned_int_exec_array;
typedef ExecutionArray<unsigned long>   unsigned_long_exec_array;
#ifdef CONDUIT_HAS_LONG_LONG
typedef ExecutionArray<unsigned long long>  unsigned_long_long_exec_array;
#endif


/// floating point arrays
typedef ExecutionArray<float>   float_exec_array;
typedef ExecutionArray<double>  double_exec_array;
#ifdef CONDUIT_USE_LONG_DOUBLE
typedef ExecutionArray<long double>  long_double_exec_array;
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
