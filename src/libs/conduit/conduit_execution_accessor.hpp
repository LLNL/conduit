// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_execution_accessor.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_EXECUTION_ACCESSOR_HPP
#define CONDUIT_EXECUTION_ACCESSOR_HPP


// TODO add setters from data arrays/accessors and exec arrays

//-----------------------------------------------------------------------------
// -- conduit  includes -- 
//-----------------------------------------------------------------------------
#include "conduit_core.hpp"
#include "conduit_data_type.hpp"
#include "conduit_utils.hpp"
#include "conduit_execution.hpp"


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- forward declarations required for conduit::ExecutionAccessor --
//-----------------------------------------------------------------------------
class Node;

//-----------------------------------------------------------------------------
// -- begin conduit::ExecutionAccessor --
//-----------------------------------------------------------------------------
///
/// class: conduit::ExecutionAccessor
///
/// description:
///  TODO
///
//-----------------------------------------------------------------------------
template <typename T> 
class CONDUIT_API ExecutionAccessor
{
public:

//-----------------------------------------------------------------------------
//
// -- conduit::ExecutionAccessor public methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Construction and Destruction
//-----------------------------------------------------------------------------
    /// Default constructor
    ExecutionAccessor();
    /// Copy constructor
    ExecutionAccessor(const ExecutionAccessor<T> &accessor);
    /// Access a pointer to node data according to node dtype description.
    ExecutionAccessor(Node &node);
    // /// Access a const pointer to node data according to node dtype description.
    // ExecutionAccessor(const Node &node);
    /// Access a pointer to node data according to node dtype description.
    ExecutionAccessor(Node *node);
    /// Access a const pointer to node data according to node dtype description.
    ExecutionAccessor(const Node *node);
    /// Destructor.
    ~ExecutionAccessor();

    ///
    /// Summary Stats Helpers
    ///
    T               min()  const;
    T               max()  const;
    T               sum()  const;
    float64         mean() const;
    
    /// counts number of occurrences of given value
    index_t         count(T value) const;

    /// Assignment operator
    ExecutionAccessor<T>   &operator=(const ExecutionAccessor<T> &accessor);

//-----------------------------------------------------------------------------
// Data and Info Access
//-----------------------------------------------------------------------------
    T              operator[](index_t idx) const
                    {return element(idx);}

    T              element(index_t idx) const;

    void           set(index_t idx, T value);

    void           fill(T value);

    const void     *element_ptr(index_t idx) const
                    {
                        return static_cast<const char*>(m_data) +
                               dtype().element_index(idx);
                    }

    index_t         number_of_elements() const 
                    {return dtype().number_of_elements();}

    const DataType &dtype() const;

    const DataType &orig_dtype() const;

    const DataType &other_dtype() const;

//-----------------------------------------------------------------------------
// Cool Stuff
//-----------------------------------------------------------------------------
    void            use_with(conduit::execution::policy policy);

    void            sync();

    void            assume();

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

    std::string     to_yaml() const;
    void            to_yaml_stream(std::ostream &os) const;

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
// -- conduit::ExecutionAccessor private data members --
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
// -- end conduit::ExecutionAccessor --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- conduit::ExecutionAccessor typedefs for supported types --
//
//-----------------------------------------------------------------------------

/// Note: these are also the types we explicitly instantiate.

/// signed integer arrays
typedef ExecutionAccessor<int8>     int8_exec_accessor;
typedef ExecutionAccessor<int16>    int16_exec_accessor;
typedef ExecutionAccessor<int32>    int32_exec_accessor;
typedef ExecutionAccessor<int64>    int64_exec_accessor;

/// unsigned integer arrays
typedef ExecutionAccessor<uint8>    uint8_exec_accessor;
typedef ExecutionAccessor<uint16>   uint16_exec_accessor;
typedef ExecutionAccessor<uint32>   uint32_exec_accessor;
typedef ExecutionAccessor<uint64>   uint64_exec_accessor;

/// floating point arrays
typedef ExecutionAccessor<float32>  float32_exec_accessor;
typedef ExecutionAccessor<float64>  float64_exec_accessor;

/// index type arrays
typedef ExecutionAccessor<index_t>  index_t_exec_accessor;

/// native c types arrays
typedef ExecutionAccessor<char>       char_exec_accessor;
typedef ExecutionAccessor<short>      short_exec_accessor;
typedef ExecutionAccessor<int>        int_exec_accessor;
typedef ExecutionAccessor<long>       long_exec_accessor;
#ifdef CONDUIT_HAS_LONG_LONG
typedef ExecutionAccessor<long long>  long_long_exec_accessor;
#endif


/// signed integer arrays
typedef ExecutionAccessor<signed char>       signed_char_exec_accessor;
typedef ExecutionAccessor<signed short>      signed_short_exec_accessor;
typedef ExecutionAccessor<signed int>        signed_int_exec_accessor;
typedef ExecutionAccessor<signed long>       signed_long_exec_accessor;
#ifdef CONDUIT_HAS_LONG_LONG
typedef ExecutionAccessor<signed long long>  signed_long_long_exec_accessor;
#endif


/// unsigned integer arrays
typedef ExecutionAccessor<unsigned char>   unsigned_char_exec_accessor;
typedef ExecutionAccessor<unsigned short>  unsigned_short_exec_accessor;
typedef ExecutionAccessor<unsigned int>    unsigned_int_exec_accessor;
typedef ExecutionAccessor<unsigned long>   unsigned_long_exec_accessor;
#ifdef CONDUIT_HAS_LONG_LONG
typedef ExecutionAccessor<unsigned long long>  unsigned_long_long_exec_accessor;
#endif


/// floating point arrays
typedef ExecutionAccessor<float>   float_exec_accessor;
typedef ExecutionAccessor<double>  double_exec_accessor;
#ifdef CONDUIT_USE_LONG_DOUBLE
typedef ExecutionAccessor<long double>  long_double_exec_accessor;
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
