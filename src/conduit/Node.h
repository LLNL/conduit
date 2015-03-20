//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//


//-----------------------------------------------------------------------------
///
/// file: Node.h
///
//-----------------------------------------------------------------------------

#ifndef __CONDUIT_NODE_H
#define __CONDUIT_NODE_H

//-----------------------------------------------------------------------------
// -- conduit library includes -- 
//-----------------------------------------------------------------------------
#include "Core.h"
#include "Error.h"
#include "Endianness.h"
#include "DataType.h"
#include "DataArray.h"
#include "Schema.h"
#include "Generator.h"
#include "NodeIterator.h"

//-----------------------------------------------------------------------------
// -- standard lib includes -- 
//-----------------------------------------------------------------------------
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- forward declarations required for conduit::Node --
//-----------------------------------------------------------------------------
class Generator;
class NodeIterator;

//-----------------------------------------------------------------------------
// -- begin conduit::Node --
//-----------------------------------------------------------------------------
///
/// class: conduit::Node
///
/// description:
///  Node is the primary class in conduit.
///
//-----------------------------------------------------------------------------
class CONDUIT_API Node
{

//=============================================================================
//-----------------------------------------------------------------------------
//
// -- public methods -- 
//
//-----------------------------------------------------------------------------
//=============================================================================
public:
    
//-----------------------------------------------------------------------------
// -- friends of Node --
//-----------------------------------------------------------------------------
    /// Note on use of `friend`: 
    ///  NodeIterator needs access to Node internals to create
    ///   an efficient iterator
    friend class NodeIterator;

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node construction and destruction --
//
//-----------------------------------------------------------------------------
///@name Construction and Destruction
///@{
//-----------------------------------------------------------------------------
/// description:
///  Standard construction and destruction methods.
///
/// notes:
///  TODO:
///  Constructors currently use a mix of copy and pointer (external) semantics
///
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- basic constructor and destruction -- 
//-----------------------------------------------------------------------------
    Node();
    Node(const Node &node);
    ~Node();

    // returns any node to the empty state
    void reset();
    
//-----------------------------------------------------------------------------
// -- constructors for generic types --
//-----------------------------------------------------------------------------
    explicit Node(const DataType &dtype);
    explicit Node(const Schema &schema);

    /// in these methods the `external` param controls if we use copy or
    /// external semantics.
    Node(const Generator &gen,
         bool external);

    Node(const std::string &json_schema,
         void *data,
         bool external);
   
    Node(const Schema &schema,
         void *data,
         bool external);

    Node(const DataType &dtype,
         void *data,
         bool external);

//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Node construction and destruction --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node generate methods --
//
//-----------------------------------------------------------------------------
///@name Generation from JSON Schemas
///@{
//-----------------------------------------------------------------------------
/// description:
///  These methods use a Generator to parse a json schema into a Node hierarchy.
///
/// * The variants without a void * data parameter will allocate memory for the
///   Node hierarchy and populate with inline values from the json schema (if
///   they are provided).
/// * The `external' variants build a Node hierarchy that points to the input
///   data, they do not copy the data into the Node hierarchy.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- direct use of a generator --
//-----------------------------------------------------------------------------
    void generate(const Generator &gen);

    void generate_external(const Generator &gen);

//-----------------------------------------------------------------------------
// -- json schema only --
//-----------------------------------------------------------------------------
    void generate(const std::string &json_schema);

    void generate(const std::string &json_schema,
                  const std::string &protocol);


//-----------------------------------------------------------------------------
// -- json schema coupled with in-core data -- 
//-----------------------------------------------------------------------------
    void generate(const std::string &json_schema,
                  void *data);

    void generate(const std::string &json_schema,
                  const std::string &protocol,
                  void *data);

    void generate_external(const std::string &json_schema,
                           void *data);

    void generate_external(const std::string &json_schema,
                           const std::string &protocol,
                           void *data);

//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Node generate methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin declaration of Node basic i/o methods --
//
//-----------------------------------------------------------------------------
///@name Binary and Memory-Mapped I/O
///@{
//-----------------------------------------------------------------------------
/// description:
///
//-----------------------------------------------------------------------------
    void load(const Schema &schema,
              const std::string &stream_path);

    /// dual file (schema + data) load
    void load(const std::string &ibase);

    void save(const std::string &obase) const; 

    void mmap(const Schema &schema,
              const std::string &stream_path);

    /// dual file (schema + data) mmap load
    void mmap(const std::string &ibase);

//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Node basic i/o methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node set methods --
//
//-----------------------------------------------------------------------------
///@name Node::set(...)
///@{
//-----------------------------------------------------------------------------
/// description:
///   set(...) methods follow copy semantics. 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for generic types --
//-----------------------------------------------------------------------------
    void set(const Node &data);
    void set(const DataType &dtype);
    void set(const Schema &schema);

    void set(const Schema &schema, void *data);
    void set(const DataType &dtype, void *data);

//-----------------------------------------------------------------------------
// -- set for scalar types ---
//-----------------------------------------------------------------------------
    // signed integer scalar types
    void set(int8 data);
    void set(int16 data);
    void set(int32 data);
    void set(int64 data);

    // unsigned integer scalar types
    void set(uint8 data);
    void set(uint16 data);
    void set(uint32 data);
    void set(uint64 data);

    // floating point scalar types
    void set(float32 data);
    void set(float64 data);

//-----------------------------------------------------------------------------
// -- set for std::vector types ---
//-----------------------------------------------------------------------------
    // signed integer array types via std::vector
    void set(const std::vector<int8>   &data);
    void set(const std::vector<int16>  &data);
    void set(const std::vector<int32>  &data);
    void set(const std::vector<int64>  &data);

    // unsigned integer array types via std::vector
    void set(const std::vector<uint8>   &data);
    void set(const std::vector<uint16>  &data);
    void set(const std::vector<uint32>  &data);
    void set(const std::vector<uint64>  &data);

    // floating point array types via std::vector
    void set(const std::vector<float32> &data);
    void set(const std::vector<float64> &data);

//-----------------------------------------------------------------------------
// -- set for conduit::DataArray types ---
//-----------------------------------------------------------------------------
    // signed integer array types via conduit::DataArray
    void set(const int8_array  &data);
    void set(const int16_array &data);
    void set(const int32_array &data);
    void set(const int64_array &data);

    // unsigned integer array types via conduit::DataArray
    void set(const uint8_array  &data);
    void set(const uint16_array &data);
    void set(const uint32_array &data);
    void set(const uint64_array &data);

    // floating point array types via conduit::DataArray
    void set(const float32_array &data);
    void set(const float64_array &data);

//-----------------------------------------------------------------------------
// -- set for string types -- 
//-----------------------------------------------------------------------------
    // char8_str use cases
    void set(const std::string &data);
    // special explicit case for string to avoid any overloading ambiguity
    void set_char8_str(const char *data);

//-----------------------------------------------------------------------------
// -- set via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
    // signed integer pointer cases
    void set(int8  *data,
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::int8),
             index_t element_bytes = sizeof(conduit::int8),
             index_t endianness = Endianness::DEFAULT_T);
    
    void set(int16 *data, 
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::int16),
             index_t element_bytes = sizeof(conduit::int16),
             index_t endianness = Endianness::DEFAULT_T);
    
    void set(int32 *data,
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::int32),
             index_t element_bytes = sizeof(conduit::int32),
             index_t endianness = Endianness::DEFAULT_T);

    void set(int64 *data,
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::int64),
             index_t element_bytes = sizeof(conduit::int64),
             index_t endianness = Endianness::DEFAULT_T);

    // unsigned integer pointer cases
    void set(uint8  *data,
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::uint8),
             index_t element_bytes = sizeof(conduit::uint8),
             index_t endianness = Endianness::DEFAULT_T);

    void set(uint16 *data,
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::uint16),
             index_t element_bytes = sizeof(conduit::uint16),
             index_t endianness = Endianness::DEFAULT_T);

    void set(uint32 *data, 
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::uint32),
             index_t element_bytes = sizeof(conduit::uint32),
             index_t endianness = Endianness::DEFAULT_T);
                      
    void set(uint64 *data,
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::uint64),
             index_t element_bytes = sizeof(conduit::uint64),
             index_t endianness = Endianness::DEFAULT_T);

    // floating point pointer cases
    void set(float32 *data,
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::float32),
             index_t element_bytes = sizeof(conduit::float32),
             index_t endianness = Endianness::DEFAULT_T);

    void set(float64 *data, 
             index_t num_elements = 1,
             index_t offset = 0,
             index_t stride = sizeof(conduit::float64),
             index_t element_bytes = sizeof(conduit::float64),
             index_t endianness = Endianness::DEFAULT_T);

//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Node set methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node set_path methods --
//
//-----------------------------------------------------------------------------
///@name Node::set_path(...)
///@{
//-----------------------------------------------------------------------------
/// description:
///   set_path(...) methods methods follow copy semantics and allow you to use
///    an explicit path for the destination.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set_path for generic types --
//-----------------------------------------------------------------------------
     void set_path(const std::string &path,
                   const Node& data);

     void set_path(const std::string &path,
                   const DataType& dtype);

     void set_path(const std::string &path,
                   const Schema &schema);
              
     void set_path(const std::string &path,
                   const Schema &schema,
                   void *data);

     void set_path(const std::string &path,
                   const DataType &dtype,
                   void *data);

//-----------------------------------------------------------------------------
// -- set_path for scalar types ---
//-----------------------------------------------------------------------------
     // signed integer scalar types
     void set_path(const std::string &path, int8 data);
     void set_path(const std::string &path, int16 data);
     void set_path(const std::string &path, int32 data);
     void set_path(const std::string &path, int64 data);

     // unsigned integer scalar types 
     void set_path(const std::string &path, uint8 data);
     void set_path(const std::string &path, uint16 data);
     void set_path(const std::string &path, uint32 data);
     void set_path(const std::string &path, uint64 data);

     // floating point scalar types
     void set_path(const std::string &path, float32 data);
     void set_path(const std::string &path, float64 data);

 //-----------------------------------------------------------------------------
 // -- set_path for std::vector types ---
 //-----------------------------------------------------------------------------
     // signed integer array types via std::vector
     void set_path(const std::string &path, const std::vector<int8>   &data);
     void set_path(const std::string &path, const std::vector<int16>  &data);
     void set_path(const std::string &path, const std::vector<int32>  &data);
     void set_path(const std::string &path, const std::vector<int64>  &data);
     
     // unsigned integer array types via std::vector
     void set_path(const std::string &path, const std::vector<uint8>   &data);
     void set_path(const std::string &path, const std::vector<uint16>  &data);
     void set_path(const std::string &path, const std::vector<uint32>  &data);
     void set_path(const std::string &path, const std::vector<uint64>  &data);
     
     // floating point array types via std::vector
     void set_path(const std::string &path, const std::vector<float32> &data);
     void set_path(const std::string &path, const std::vector<float64> &data);

 //-----------------------------------------------------------------------------
 // -- set_path for conduit::DataArray types ---
 //-----------------------------------------------------------------------------
     // signed integer array types via conduit::DataArray
     void set_path(const std::string &path, const int8_array  &data);
     void set_path(const std::string &path, const int16_array &data);
     void set_path(const std::string &path, const int32_array &data);
     void set_path(const std::string &path, const int64_array &data);

     // unsigned integer array types via conduit::DataArray
     void set_path(const std::string &path, const uint8_array  &data);
     void set_path(const std::string &path, const uint16_array &data);
     void set_path(const std::string &path, const uint32_array &data);
     void set_path(const std::string &path, const uint64_array &data);

     // floating point array types via conduit::DataArray
     void set_path(const std::string &path, const float32_array &data);
     void set_path(const std::string &path, const float64_array &data);

//-----------------------------------------------------------------------------
// -- set_path for string types -- 
//-----------------------------------------------------------------------------
     // char8_str use cases
     void set_path(const std::string &path,
                   const std::string &data);

     // special explicit case for string to avoid any overloading ambiguity
     void set_path_char8_str(const std::string &path,
                             const char* data);

//-----------------------------------------------------------------------------
// -- set_path via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------  // signed integer pointer cases
     void set_path(const std::string &path,int8  *data,
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::int8),
                   index_t element_bytes = sizeof(conduit::int8),
                   index_t endianness = Endianness::DEFAULT_T);
        
     void set_path(const std::string &path,
                   int16 *data, 
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::int16),
                   index_t element_bytes = sizeof(conduit::int16),
                   index_t endianness = Endianness::DEFAULT_T);

     void set_path(const std::string &path,
                   int32 *data,
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::int32),
                   index_t element_bytes = sizeof(conduit::int32),
                   index_t endianness = Endianness::DEFAULT_T);

     void set_path(const std::string &path,
                   int64 *data,
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::int64),
                   index_t element_bytes = sizeof(conduit::int64),
                   index_t endianness = Endianness::DEFAULT_T);

    // unsigned integer pointer cases
     void set_path(const std::string &path,
                   uint8  *data,
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::uint8),
                   index_t element_bytes = sizeof(conduit::uint8),
                   index_t endianness = Endianness::DEFAULT_T);

     void set_path(const std::string &path,
                   uint16 *data,
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::uint16),
                   index_t element_bytes = sizeof(conduit::uint16),
                   index_t endianness = Endianness::DEFAULT_T);

     void set_path(const std::string &path,
                   uint32 *data, 
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::uint32),
                   index_t element_bytes = sizeof(conduit::uint32),
                   index_t endianness = Endianness::DEFAULT_T);
              
     void set_path(const std::string &path,
                   uint64 *data,
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::uint64),
                   index_t element_bytes = sizeof(conduit::uint64),
                   index_t endianness = Endianness::DEFAULT_T);

     // floating point integer pointer cases
     void set_path(const std::string &path,
                   float32 *data,
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::float32),
                   index_t element_bytes = sizeof(conduit::float32),
                   index_t endianness = Endianness::DEFAULT_T);

     void set_path(const std::string &path,
                   float64 *data, 
                   index_t num_elements = 1,
                   index_t offset = 0,
                   index_t stride = sizeof(conduit::float64),
                   index_t element_bytes = sizeof(conduit::float64),
                   index_t endianness = Endianness::DEFAULT_T);

//-----------------------------------------------------------------------------
///@}                      
//-----------------------------------------------------------------------------
//
// -- end declaration of Node set_path methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node set_external methods --
//
//-----------------------------------------------------------------------------
///@name Node::set_external(...)
///@{
//-----------------------------------------------------------------------------
/// description:
///   set_external(...) methods methods follow pointer semantics.
///   (they do not copy data into the node, but point to the data passed)
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set for generic types --
//-----------------------------------------------------------------------------

    void    set_external(Node &n);
    void    set_external(const Schema &schema, void *data);
    void    set_external(const DataType &dtype, void *data);

//-----------------------------------------------------------------------------
// -- set_external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
    // signed integer pointer cases
    void set_external(int8  *data,
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::int8),
                      index_t element_bytes = sizeof(conduit::int8),
                      index_t endianness = Endianness::DEFAULT_T);
    
    void set_external(int16 *data, 
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::int16),
                      index_t element_bytes = sizeof(conduit::int16),
                      index_t endianness = Endianness::DEFAULT_T);
    
    void set_external(int32 *data,
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::int32),
                      index_t element_bytes = sizeof(conduit::int32),
                      index_t endianness = Endianness::DEFAULT_T);

    void set_external(int64 *data,
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::int64),
                      index_t element_bytes = sizeof(conduit::int64),
                      index_t endianness = Endianness::DEFAULT_T);

    // unsigned integer pointer cases
    void set_external(uint8  *data,
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::uint8),
                      index_t element_bytes = sizeof(conduit::uint8),
                      index_t endianness = Endianness::DEFAULT_T);

    void set_external(uint16 *data,
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::uint16),
                      index_t element_bytes = sizeof(conduit::uint16),
                      index_t endianness = Endianness::DEFAULT_T);

    void set_external(uint32 *data, 
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::uint32),
                      index_t element_bytes = sizeof(conduit::uint32),
                      index_t endianness = Endianness::DEFAULT_T);
                      
    void set_external(uint64 *data,
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::uint64),
                      index_t element_bytes = sizeof(conduit::uint64),
                      index_t endianness = Endianness::DEFAULT_T);

    // floating point pointer cases
    void set_external(float32 *data,
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::float32),
                      index_t element_bytes = sizeof(conduit::float32),
                      index_t endianness = Endianness::DEFAULT_T);

    void set_external(float64 *data, 
                      index_t num_elements = 1,
                      index_t offset = 0,
                      index_t stride = sizeof(conduit::float64),
                      index_t element_bytes = sizeof(conduit::float64),
                      index_t endianness = Endianness::DEFAULT_T);

//-----------------------------------------------------------------------------
// -- set_external for std::vector types ---
//----------------------------------------------------------------------------- // signed integer array types via std::vector
    void set_external(std::vector<int8>   &data);
    void set_external(std::vector<int16>  &data);
    void set_external(std::vector<int32>  &data);
    void set_external(std::vector<int64>  &data);

    // unsigned integer array types via std::vector
    void set_external(std::vector<uint8>   &data);
    void set_external(std::vector<uint16>  &data);
    void set_external(std::vector<uint32>  &data);
    void set_external(std::vector<uint64>  &data);

    // floating point array types via std::vector
    void set_external(std::vector<float32> &data);
    void set_external(std::vector<float64> &data);

    //-----------------------------------------------------------------------------
// -- set_external for conduit::DataArray types ---
//-----------------------------------------------------------------------------
    // signed integer array types via conduit::DataArray
    void set_external(const int8_array  &data);
    void set_external(const int16_array &data);
    void set_external(const int32_array &data);
    void set_external(const int64_array &data);

    // unsigned integer array types via conduit::DataArray
    void set_external(const uint8_array  &data);
    void set_external(const uint16_array &data);
    void set_external(const uint32_array &data);
    void set_external(const uint64_array &data);

    // floating point array types via conduit::DataArray
    void set_external(const float32_array &data);
    void set_external(const float64_array &data);

    //-----------------------------------------------------------------------------
// -- set_external for string types ---
//-----------------------------------------------------------------------------
    void set_external_char8_str(char *data);

//-----------------------------------------------------------------------------
///@}                      
//-----------------------------------------------------------------------------
//
// -- end  declaration of Node set_external methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node set_path_external methods --
//
//-----------------------------------------------------------------------------
///@name Node::set_path_external(...)
///@{
//-----------------------------------------------------------------------------
/// description:
///   set_path_external(...) methods allow the node to point to external
///   memory, and allow you to use an explicit path for the destination node.
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for generic types --
//-----------------------------------------------------------------------------
    /// TODO: set_path_external(const Node &n)
    void    set_path_external(const std::string &path,
                              const Schema &schema,
                              void *data);

    void    set_path_external(const std::string &path,
                              const DataType &dtype,
                              void *data);

//-----------------------------------------------------------------------------
// -- set_path_external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
    // signed integer pointer cases
    void set_path_external(const std::string &path,
                           int8  *data,
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::int8),
                           index_t element_bytes = sizeof(conduit::int8),
                           index_t endianness = Endianness::DEFAULT_T);


    void set_path_external(const std::string &path,
                           int16 *data, 
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::int16),
                           index_t element_bytes = sizeof(conduit::int16),
                           index_t endianness = Endianness::DEFAULT_T);
 
    void set_path_external(const std::string &path,
                           int32 *data,
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::int32),
                           index_t element_bytes = sizeof(conduit::int32),
                           index_t endianness = Endianness::DEFAULT_T);

    void set_path_external(const std::string &path,
                           int64 *data,
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::int64),
                           index_t element_bytes = sizeof(conduit::int64),
                           index_t endianness = Endianness::DEFAULT_T);

    // unsigned integer pointer cases
    void set_path_external(const std::string &path,
                           uint8  *data,
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::uint8),
                           index_t element_bytes = sizeof(conduit::uint8),
                           index_t endianness = Endianness::DEFAULT_T);


    void set_path_external(const std::string &path,
                           uint16 *data,
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::uint16),
                           index_t element_bytes = sizeof(conduit::uint16),
                           index_t endianness = Endianness::DEFAULT_T);
                           
    void set_path_external(const std::string &path,
                           uint32 *data, 
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::uint32),
                           index_t element_bytes = sizeof(conduit::uint32),
                           index_t endianness = Endianness::DEFAULT_T);

                           
    void set_path_external(const std::string &path,
                           uint64 *data,
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::uint64),
                           index_t element_bytes = sizeof(conduit::uint64),
                           index_t endianness = Endianness::DEFAULT_T);

    // floating point pointer cases
    void set_path_external(const std::string &path,
                           float32 *data,
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::float32),
                           index_t element_bytes = sizeof(conduit::float32),
                           index_t endianness = Endianness::DEFAULT_T);


    void set_path_external(const std::string &path,
                           float64 *data, 
                           index_t num_elements = 1,
                           index_t offset = 0,
                           index_t stride = sizeof(conduit::float64),
                           index_t element_bytes = sizeof(conduit::float64),
                           index_t endianness = Endianness::DEFAULT_T);

//-----------------------------------------------------------------------------
// -- set_path_external for std::vector types ---
//-----------------------------------------------------------------------------
    // signed integer array types via std::vector
    void set_path_external(const std::string &path, std::vector<int8> &data);
    void set_path_external(const std::string &path, std::vector<int16> &data);
    void set_path_external(const std::string &path, std::vector<int32> &data);
    void set_path_external(const std::string &path, std::vector<int64> &data);

    // unsigned integer array types via std::vector
    void set_path_external(const std::string &path, std::vector<uint8> &data);
    void set_path_external(const std::string &path, std::vector<uint16> &data);
    void set_path_external(const std::string &path, std::vector<uint32> &data);
    void set_path_external(const std::string &path, std::vector<uint64> &data);

    // floating point array types via std::vector
    void set_path_external(const std::string &path,
                           std::vector<float32> &data);
    void set_path_external(const std::string &path,
                           std::vector<float64> &data);
    //-----------------------------------------------------------------------------
// -- set_path_external for conduit::DataArray types ---
//-----------------------------------------------------------------------------
    // signed integer array types via conduit::DataArray
    void set_path_external(const std::string &path, const int8_array  &data);
    void set_path_external(const std::string &path, const int16_array &data);
    void set_path_external(const std::string &path, const int32_array &data);
    void set_path_external(const std::string &path, const int64_array &data);

    // unsigned integer array types via conduit::DataArray
    void set_path_external(const std::string &path, const uint8_array  &data);
    void set_path_external(const std::string &path, const uint16_array &data);
    void set_path_external(const std::string &path, const uint32_array &data);
    void set_path_external(const std::string &path, const uint64_array &data);

    // floating point array types via conduit::DataArray
    void set_path_external(const std::string &path, const float32_array &data);
    void set_path_external(const std::string &path, const float64_array &data);

//-----------------------------------------------------------------------------
// -- set_external for string types ---
//-----------------------------------------------------------------------------
    void set_path_external_char8_str(const std::string &path, char *data);

//-----------------------------------------------------------------------------
///@}                      
//-----------------------------------------------------------------------------
//
// -- end declaration of Node set_path_external methods --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin declaration of Node assignment operators --
//
//-----------------------------------------------------------------------------
///@name Node Assignment Operators
///@{
//-----------------------------------------------------------------------------
/// description:
/// &operator=(...) methods use set(...) (copy) semantics
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- assignment operators for generic types --
//-----------------------------------------------------------------------------
    Node &operator=(const Node &node);
    Node &operator=(const DataType &dtype);
    Node &operator=(const Schema  &schema);

//-----------------------------------------------------------------------------
// --  assignment operators for scalar types ---
//-----------------------------------------------------------------------------
     // signed integer scalar types
    Node &operator=(int8 data);
    Node &operator=(int16 data);
    Node &operator=(int32 data);
    Node &operator=(int64 data);

     // unsigned integer scalar types
    Node &operator=(uint8 data);
    Node &operator=(uint16 data);
    Node &operator=(uint32 data);
    Node &operator=(uint64 data);
    
    // floating point scalar types
    Node &operator=(float32 data);    
    Node &operator=(float64 data);

//-----------------------------------------------------------------------------
// -- assignment operators for std::vector types ---
//-----------------------------------------------------------------------------

    // signed integer array types via std::vector
    Node &operator=(const std::vector<int8>   &data);
    Node &operator=(const std::vector<int16>   &data);
    Node &operator=(const std::vector<int32>   &data);
    Node &operator=(const std::vector<int64>   &data);

    // unsigned integer array types via std::vector
    Node &operator=(const std::vector<uint8>   &data);
    Node &operator=(const std::vector<uint16>   &data);
    Node &operator=(const std::vector<uint32>   &data);
    Node &operator=(const std::vector<uint64>   &data);

    // floating point array types via std::vector
    Node &operator=(const std::vector<float32>  &data);
    Node &operator=(const std::vector<float64>  &data);

//-----------------------------------------------------------------------------
// -- assignment operators for conduit::DataArray types ---
//-----------------------------------------------------------------------------
    // signed integer array types via conduit::DataArray
    Node &operator=(const int8_array  &data);
    Node &operator=(const int16_array &data);
    Node &operator=(const int32_array &data);
    Node &operator=(const int64_array &data);

    // unsigned integer array ttypes via conduit::DataArray
    Node &operator=(const uint8_array  &data);
    Node &operator=(const uint16_array &data);
    Node &operator=(const uint32_array &data);
    Node &operator=(const uint64_array &data);

    // floating point array types via conduit::DataArray
    Node &operator=(const float32_array &data);
    Node &operator=(const float64_array &data);

//-----------------------------------------------------------------------------
// -- assignment operators for string types -- 
//-----------------------------------------------------------------------------
    // char8_str use cases
    Node &operator=(const char *data);
    Node &operator=(const std::string &data);

//-----------------------------------------------------------------------------
///@}                      
//-----------------------------------------------------------------------------
//
// -- end declaration of Node assignment operators --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin declaration of Node transforms --
//
//-----------------------------------------------------------------------------
///@name Node Transforms
///@{
//-----------------------------------------------------------------------------
/// description:
///  TODO
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- serialization methods ---
//-----------------------------------------------------------------------------
    /// serialize to a byte vector
    void        serialize(std::vector<uint8> &data) const;
    /// serialize to a file identified by stream_path
    void        serialize(const std::string &stream_path) const;
    /// serialize to an output stream
    void        serialize(std::ofstream &ofs) const;

//-----------------------------------------------------------------------------
// -- compaction methods ---
//-----------------------------------------------------------------------------
    /// compact this node
    void        compact();

    /// compact into a new node
    void        compact_to(Node &n_dest) const;

    /// compact and return the result
    /// TODO: this is inefficient w/o move semantics, but is very 
    /// convenient for testing and example programs.
    Node        compact_to() const;

//-----------------------------------------------------------------------------
// -- update methods ---
//-----------------------------------------------------------------------------
    /// update() adds children from n_src to current Node (analogous to a 
    /// python dictionary update) 
    ///
    /// NOTE: The input should be const, but the lack of a const fetch prevents
    /// this for now.
    void        update(Node &n_src);
    /// TODO: update_external?



//-----------------------------------------------------------------------------
// -- leaf coercion methods ---
//-----------------------------------------------------------------------------
    ///
    /// These methods allow you to coerce a leaf type to the widest bitwidth
    /// type.
    ///

    /// convert to a 64-bit signed integer 
    int64            to_int64()   const;
    /// convert to a 64-bit unsigned integer 
    uint64           to_uint64()  const;
    /// convert to a 64-bit floating point number
    float64          to_float64() const;
    /// convert to the index type 
    index_t          to_index_t() const;

//-----------------------------------------------------------------------------
// -- JSON construction methods ---
//-----------------------------------------------------------------------------

    // the generic to_json methods are used by the specialized cases 
    std::string         to_json(bool detailed=true, 
                                index_t indent=2, 
                                index_t depth=0,
                                const std::string &pad=" ",
                                const std::string &eoe="\n") const;

    void                to_json(std::ostringstream &oss,
                                bool detailed=true, 
                                index_t indent=2, 
                                index_t depth=0,
                                const std::string &pad=" ",
                                const std::string &eoe="\n") const;

    // transforms the node to json without any conduit schema constructs
    std::string      to_pure_json(index_t indent=2) const;

    void             to_pure_json(std::ostringstream &oss,
                                  index_t indent=2) const;

    // transforms the node to json that contains conduit schema constructs
    std::string      to_detailed_json(index_t indent=2, 
                                      index_t depth=0,
                                      const std::string &pad=" ",
                                      const std::string &eoe="\n") const;

    void             to_detailed_json(std::ostringstream &oss,
                                      index_t indent=2, 
                                      index_t depth=0,
                                      const std::string &pad=" ",
                                      const std::string &eoe="\n") const;

//-----------------------------------------------------------------------------
//
// -- end declaration of Node transforms --
//
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
//
// -- begin declaration of Node information methods --
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
///@name Node Information
///@{
//-----------------------------------------------------------------------------
/// description:
///  These methods provide general info about the node hierarchy, and memory 
///  layout.
//-----------------------------------------------------------------------------
    // schema access
    const Schema     &schema() const 
                        { return *m_schema;}   
    const DataType   &dtype() const       
                        { return m_schema->dtype();}


    // parent access
    bool             has_parent() const 
                        {return m_parent != NULL;}
    Node            *parent() 
                        {return m_parent;}
    
    //memory space info
    index_t          total_bytes() const 
                        { return m_schema->total_bytes();}
    index_t          total_bytes_compact() const
                        { return m_schema->total_bytes_compact();}

    /// is this node using a compact data layout
    bool             is_compact() const 
                        {return dtype().is_compact();}
    ///
    /// info() creates a node that contains metadata about the current
    /// node's memory properties
    void            info(Node &nres) const;
    /// TODO: this is inefficient w/o move semantics, but is very 
    /// convenient for testing and example programs.
    Node            info() const;

    /// TODO: compare or operator== ?

//-----------------------------------------------------------------------------
// -- stdout print methods ---
//-----------------------------------------------------------------------------
    /// print a simplified json representation of the this node to std out
    void            print(bool detailed=false) const
                        {std::cout << to_json(detailed,2) << std::endl;}
    /// print a detailed json representation of the this node to std out.
    /// json output includes conduit schema constructs
    void            print_detailed() const
                        {print(true);}
//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Node information methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node entry access methods --
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
///@name Node Entry Access Methods
///@{
//-----------------------------------------------------------------------------
/// description:
///  Node traversal (iterators), child access (for list or object types)
//-----------------------------------------------------------------------------
    /// iterator access
    NodeIterator     iterator();
    
    /// `fetch' methods do modify map structure if a path doesn't exist
    /// fetch the node at the given path
    Node             &fetch(const std::string &path);
    /// fetch the node at the given index
    Node             &child(index_t idx);

    /// fetch a pointer to the node  at the given path   
    Node             *fetch_pointer(const std::string &path);
    /// fetch a pointer to the node at the given index
    Node             *child_pointer(index_t idx);

    /// access child node via a path (equivalent to fetch via path)
    Node             &operator[](const std::string &path);
    /// access child node via index (equivalent to fetch via index)
    Node             &operator[](index_t idx);

    /// return the number of children (list and object interfaces)
    index_t number_of_children() const;
    
    /// checks if given path exists in the Node hierarchy 
    bool    has_path(const std::string &path) const;
    /// returns the direct child paths for this node
    void    paths(std::vector<std::string> &paths) const;

    /// adds an empty unnamed node to a list (list interface)
    /// TODO `append` is a strange name here, we want this interface
    /// but we may be abusing the common concept folks think of
    //  for the term `append`.
    Node   &append();

    /// remove child at index (list and object interfaces)
    void    remove(index_t idx);
    /// remove child at given path (object interface)
    void    remove(const std::string &path);


//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Node entry access methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Node value access methods --
//
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
///@name Node Value Access Methods
///@{
//-----------------------------------------------------------------------------
/// description:
///  Direct access to data at leaf types.
//-----------------------------------------------------------------------------
     
     // signed integer scalars
    int8             as_int8()   const  
                        { return *((int8*)element_pointer(0));}
    int16            as_int16()  const  
                        { return *((int16*)element_pointer(0));}
    int32            as_int32()  const  
                        { return *((int32*)element_pointer(0));}
    int64            as_int64()  const  
                        { return *((int64*)element_pointer(0));}

    // unsigned integer scalars
    uint8            as_uint8()   const 
                        { return *((uint8*)element_pointer(0));}
    uint16           as_uint16()  const 
                        { return *((uint16*)element_pointer(0));}
    uint32           as_uint32()  const 
                        { return *((uint32*)element_pointer(0));}
    uint64           as_uint64()  const 
                        { return *((uint64*)element_pointer(0));}

    // floating point scalars
    float32          as_float32() const 
                        { return *((float32*)element_pointer(0));}
    float64          as_float64() const 
                        { return *((float64*)element_pointer(0));}

    // signed integers via pointers
    int8            *as_int8_ptr()     
                        { return (int8*)element_pointer(0);}
    int16           *as_int16_ptr()    
                        { return (int16*)element_pointer(0);}
    int32           *as_int32_ptr()    
                        { return (int32*)element_pointer(0);}
    int64           *as_int64_ptr()    
                        { return (int64*)element_pointer(0);}

    // unsigned integers via pointers
    uint8           *as_uint8_ptr()    
                        { return (uint8*)element_pointer(0);}
    uint16          *as_uint16_ptr()   
                        { return (uint16*)element_pointer(0);}
    uint32          *as_uint32_ptr()   
                        { return (uint32*)element_pointer(0);}
    uint64          *as_uint64_ptr()   
                        { return (uint64*)element_pointer(0);}

    // floating point via pointers
    float32         *as_float32_ptr()  
                        { return (float32*)element_pointer(0);}
    float64         *as_float64_ptr()  
                        { return (float64*)element_pointer(0);}

    // signed integer array types via conduit::DataArray
    int8_array       as_int8_array()   
                        { return int8_array(m_data,dtype());}
    int16_array      as_int16_array()  
                        { return int16_array(m_data,dtype());}
    int32_array      as_int32_array()  
                        { return int32_array(m_data,dtype());}
    int64_array      as_int64_array()  
                        { return int64_array(m_data,dtype());}

    // unsigned integer array types via conduit::DataArray
    uint8_array      as_uint8_array()  
                        { return uint8_array(m_data,dtype());}
    uint16_array     as_uint16_array() 
                        { return uint16_array(m_data,dtype());}
    uint32_array     as_uint32_array() 
                        { return uint32_array(m_data,dtype());}
    uint64_array     as_uint64_array() 
                        { return uint64_array(m_data,dtype());}

    // floating point array types via conduit::DataArray
    float32_array    as_float32_array() 
                        { return float32_array(m_data,dtype());}
    float64_array    as_float64_array() 
                        { return float64_array(m_data,dtype());}

    // signed integer array types via conduit::DataArray (const variants)

    int8_array       as_int8_array()  const 
                        { return int8_array(m_data,dtype());}
    int16_array      as_int16_array() const 
                        { return int16_array(m_data,dtype());}
    int32_array      as_int32_array() const 
                        { return int32_array(m_data,dtype());}
    int64_array      as_int64_array() const 
                        { return int64_array(m_data,dtype());}

    // unsigned integer array types via conduit::DataArray (const variants)
    uint8_array      as_uint8_array()  const 
                        { return uint8_array(m_data,dtype());}
    uint16_array     as_uint16_array() const 
                        { return uint16_array(m_data,dtype());}
    uint32_array     as_uint32_array() const 
                        { return uint32_array(m_data,dtype());}
    uint64_array     as_uint64_array() const 
                        { return uint64_array(m_data,dtype());}

    // floating point array value via conduit::DataArray (const variants)
    float32_array    as_float32_array() const 
                        { return float32_array(m_data,dtype());}
    float64_array    as_float64_array() const 
                        { return float64_array(m_data,dtype());}

    // char8_str cases
    char            *as_char8_str() 
                        {return (char *)element_pointer(0);}
    const char      *as_char8_str() const 
                        {return (const char *)element_pointer(0);}
    
    std::string      as_string() const 
                        {return std::string(as_char8_str());}

    // direct data pointer access 
    uint8            *data_pointer() 
                        {return (uint8*)m_data;}


//-----------------------------------------------------------------------------
/// description:
///  Direct access to data at leaf types (native c++ types)
//-----------------------------------------------------------------------------
     
     // signed integer scalars
    char           as_char()  const
                   { return *((char*)element_pointer(0));}

    short          as_short()  const
                   { return *((short*)element_pointer(0));}

    int            as_int()  const  
                   { return *((int*)element_pointer(0));}

    long           as_long()  const  
                   { return *((long*)element_pointer(0));}

    // unsigned integer scalars
    unsigned char   as_unsigned_char()   const 
                        { return *((unsigned char*)element_pointer(0));}

    unsigned short   as_unsigned_short()   const 
                        { return *((unsigned short*)element_pointer(0));}
    
    unsigned int     as_unsigned_int()   const 
                        { return *((unsigned int*)element_pointer(0));}
    unsigned long    as_unsigned_long()  const 
                        { return *(( unsigned long*)element_pointer(0));}

    // floating point scalars
    float            as_float() const 
                       { return *((float*)element_pointer(0));}
    double           as_double() const 
                        { return *((double*)element_pointer(0));}

    // signed integers via pointers
    
    char            *as_char_ptr()     
                        { return (char*)element_pointer(0);}
    short           *as_short_ptr()    
                        { return (short*)element_pointer(0);}
    int             *as_int_ptr()    
                        { return (int*)element_pointer(0);}
    long            *as_long_ptr()    
                        { return (long*)element_pointer(0);}

    // unsigned integers via pointers
    unsigned char      *as_unsigned_char_ptr()    
                        { return (unsigned char*)element_pointer(0);}
    unsigned short      *as_unsigned_short_ptr()   
                       { return (unsigned short*)element_pointer(0);}
    unsigned int       *as_unsigned_int_ptr()   
                        { return (unsigned int*)element_pointer(0);}
    unsigned long      *as_unsigned_long_ptr()   
                        { return (unsigned long*)element_pointer(0);}

    // floating point via pointers
    float             *as_float_ptr()  
                        { return (float*)element_pointer(0);}
    double           *as_double_ptr()  
                        { return (double*)element_pointer(0);}

    // signed integer array types via conduit::DataArray
    char_array       as_char_array()
                        { return char_array(m_data,dtype());}
    short_array      as_short_array()
                        { return short_array(m_data,dtype());}
    int_array        as_int_array()
                        { return int_array(m_data,dtype());}
    long_array      as_long_array()
                        { return long_array(m_data,dtype());}

    // unsigned integer array types via conduit::DataArray
    unsigned_char_array    as_unsigned_char_array()
                            { return unsigned_char_array(m_data,dtype());}
    unsigned_short_array   as_unsigned_short_array()
                            { return unsigned_short_array(m_data,dtype());}
    unsigned_int_array     as_unsigned_int_array()
                            { return unsigned_int_array(m_data,dtype());}
    unsigned_long_array    as_unsigned_long_()
                            { return unsigned_long_array(m_data,dtype());}

    // floating point array types via conduit::DataArray
    float_array     as_float_array()
                        { return float_array(m_data,dtype());}
    double_array    as_double_array()
                        { return double_array(m_data,dtype());}

    // signed integer array types via conduit::DataArray (const variants)

    char_array       as_char_array() const
                        { return char_array(m_data,dtype());}
    short_array      as_short_array() const
                        { return short_array(m_data,dtype());}
    int_array        as_int_array() const
                        { return int_array(m_data,dtype());}
    long_array      as_long_array() const
                        { return long_array(m_data,dtype());}

    // unsigned integer array types via conduit::DataArray (const variants)
    unsigned_char_array    as_unsigned_char_array() const
                            { return unsigned_char_array(m_data,dtype());}
    unsigned_short_array   as_unsigned_short_array() const
                            { return unsigned_short_array(m_data,dtype());}
    unsigned_int_array     as_unsigned_int_array() const
                            { return unsigned_int_array(m_data,dtype());}
    unsigned_long_array    as_unsigned_long_() const
                            { return unsigned_long_array(m_data,dtype());}


    // floating point array value via conduit::DataArray (const variants)
    float_array     as_float_array() const
                        { return float_array(m_data,dtype());}
    double_array    as_double_array() const
                        { return double_array(m_data,dtype());}

//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Node value access methods --
//
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//
// -- begin declaration of Interface Warts --
//
//-----------------------------------------------------------------------------
///@name Interface Warts 
///@{
//-----------------------------------------------------------------------------
/// description:
/// these methods are used for construction by the Node & Generator classes
/// it would be nice to make them private, however to keep rapid json
/// headers out of the conduit interface, the json walking methods
/// aren't part of any public interface. We haven't found the right way
/// to use 'friend' to avoid this issue
//-----------------------------------------------------------------------------
    void             set_data_pointer(void *data_ptr);
    void             set_schema_pointer(Schema *schema_ptr);
    void             append_node_pointer(Node *node)
                        {m_children.push_back(node);}
    void             set_parent(Node *parent) 
                        { m_parent = parent;}
    Schema          *schema_pointer() 
                        {return m_schema;}
//-----------------------------------------------------------------------------
///@}
//-----------------------------------------------------------------------------
//
// -- end declaration of Interface Warts --
//
//-----------------------------------------------------------------------------


//=============================================================================
//-----------------------------------------------------------------------------
//
// -- private methods and members -- 
//
//-----------------------------------------------------------------------------
//=============================================================================

private:
//-----------------------------------------------------------------------------
//
// -- private methods that help with init, memory allocation, and cleanup --
//
//-----------------------------------------------------------------------------
    // setup a node to at as a given type
    void             init(const DataType &dtype);
    // memory allocation and mapping routines
    void             allocate(index_t dsize);
    void             allocate(const DataType &dtype);
    void             mmap(const std::string &stream_path,
                          index_t dsize);
    // release any alloced or memory mapped data
    void             release();
    // clean up everything (used by destructor)
    void             cleanup();

    // set defaults (used by constructors)
    void              init_defaults();
    // setup node to act as a list
    void              init_list();
    // setup node to act as an object
    void              init_object();

//-----------------------------------------------------------------------------
//
// -- private methods that help with hierarchical construction --
//
//-----------------------------------------------------------------------------
    // work horse for complex node hierarchical setup
    static void      walk_schema(Node   *node,
                                 Schema *schema,
                                 void   *data);

    static void      mirror_node(Node   *node,
                                 Schema *schema,
                                 Node   *src);

//-----------------------------------------------------------------------------
//
// -- private methods that help element access -- 
//
//-----------------------------------------------------------------------------
          void  *element_pointer(index_t idx)
        {return static_cast<char*>(m_data) + dtype().element_index(idx);};
    const void  *element_pointer(index_t idx) const 
        {return static_cast<char*>(m_data) + dtype().element_index(idx);};


//-----------------------------------------------------------------------------
//
// -- private methods that help with compaction, serialization, and info  --
//
//-----------------------------------------------------------------------------
    void              compact_to(uint8 *data,
                                 index_t curr_offset) const;
    /// compact helper for leaf types
    void              compact_elements_to(uint8 *data) const;


    void              serialize(uint8 *data,
                                index_t curr_offset) const;

    void              info(Node &res,
                           const std::string &curr_path) const;

//-----------------------------------------------------------------------------
//
// -- conduit::Node private data members --
//
//-----------------------------------------------------------------------------
    /// pointer to this node's parent (if it exists)
    Node                *m_parent;
    /// pointer to this node's schema
    Schema              *m_schema;
    /// collection of children
    std::vector<Node*>   m_children;

    // TODO: DataContainer?
    // pointer to the node's data
    void     *m_data;
    // size of the allocated or mmaped data point
    index_t   m_data_size;

    // flag that indicates this node allocated m_data
    bool      m_alloced;
    // flag that indicates if m_data is memory-mapped
    bool      m_mmaped;
    // memory-map file descriptor
    int       m_mmap_fd;
};

//-----------------------------------------------------------------------------
// -- end conduit::Node --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------

#endif
