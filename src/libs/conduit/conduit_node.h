//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://scalability-llnl.github.io/conduit/.
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
/// file: conduit_node.h
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_NODE_H
#define CONDUIT_NODE_H

#include <stdlib.h>
#include <stddef.h>
    
#include "Bitwidth_Style_Types.h"
#include "Endianness_Types.h"

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------
#ifdef __cplusplus
extern "C" {
#endif

//-----------------------------------------------------------------------------
// -- typedef for conduit_node --
//-----------------------------------------------------------------------------

typedef void  conduit_node;

//-----------------------------------------------------------------------------
// -- conduit_node creation and destruction --
//-----------------------------------------------------------------------------

conduit_node *conduit_node_create();
void          conduit_node_destroy(conduit_node *cnode);


//-----------------------------------------------------------------------------
// -- object and list interface methods -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
conduit_node *conduit_node_fetch(conduit_node *cnode,
                                 const char *path);

//-----------------------------------------------------------------------------
conduit_node *conduit_node_append(conduit_node *cnode);

//-----------------------------------------------------------------------------
conduit_node *conduit_node_child(conduit_node *cnode,
                                 size_t idx);

//-----------------------------------------------------------------------------
size_t        conduit_node_number_of_children(conduit_node *cnode);

//-----------------------------------------------------------------------------
size_t        conduit_node_number_of_elements(conduit_node *cnode);

//-----------------------------------------------------------------------------
// -- node info -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int           conduit_node_is_root(conduit_node *cnode);

//-----------------------------------------------------------------------------
void          conduit_node_print(conduit_node *cnode);
void          conduit_node_print_detailed(conduit_node *cnode);

//-----------------------------------------------------------------------------
// -- set for generic types --
//-----------------------------------------------------------------------------
    void conduit_set_node(conduit_node *cnode,
                          conduit_node *data);

    // TODO: These req c-interfaces for datatype, schema, etc
    //void set_dtype(const DataType &dtype);
    //void set_schema(const Schema &schema);
    //void set_data_using_dtype(const DataType &dtype, void *data);
    //void set_data_using_schema(const Schema &schema, void *data);

//-----------------------------------------------------------------------------
// -- set for scalar bitwidth style types ---
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // signed integer scalar types
    //-------------------------------------------------------------------------
    void          conduit_node_set_int8(conduit_node *cnode,
                                        conduit_int8 value);
                                     
    void          conduit_node_set_int16(conduit_node *cnode,
                                         conduit_int16 value);

    void          conduit_node_set_int32(conduit_node *cnode,
                                         conduit_int32 value);

    void          conduit_node_set_int64(conduit_node *cnode,
                                         conduit_int64 value);
   //-------------------------------------------------------------------------
   // unsigned integer scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_uint8(conduit_node *cnode,
                                         conduit_uint8 value);
                                     
    void          conduit_node_set_uint16(conduit_node *cnode,
                                          conduit_uint16 value);

    void          conduit_node_set_uint32(conduit_node *cnode,
                                          conduit_uint32 value);

    void          conduit_node_set_uint64(conduit_node *cnode,
                                          conduit_uint64 value);

   //-------------------------------------------------------------------------
   // floating point scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_float32(conduit_node *cnode,
                                           conduit_float32 value);

    void          conduit_node_set_float64(conduit_node *cnode,
                                           conduit_float64 value);

   //-------------------------------------------------------------------------
   // string cases
   //-------------------------------------------------------------------------
   void           conduit_node_set_char8_str(conduit_node *cnode, 
                                             const char *value);

// //-----------------------------------------------------------------------------
// // -- set for scalar c-style style types ---
// //-----------------------------------------------------------------------------
//     //-------------------------------------------------------------------------
//     // signed integer scalar types
//     //-------------------------------------------------------------------------
//     void          conduit_node_set_char(conduit_node *cnode,
//                                         char value);
//
//     void          conduit_node_set_short(conduit_node *cnode,
//                                          short value);
//
//     void          conduit_node_set_int(conduit_node *cnode,
//                                        int value);
//
//     void          conduit_node_set_long(conduit_node *cnode,
//                                         long value);
//
//    //-------------------------------------------------------------------------
//    // unsigned integer scalar types
//    //-------------------------------------------------------------------------
//     void          conduit_node_set_unsigned_char(conduit_node *cnode,
//                                                  unsigned char value);
//
//     void          conduit_node_set_unsigned_short(conduit_node *cnode,
//                                                   unsigned short value);
//
//     void          conduit_node_set_unsigned_int(conduit_node *cnode,
//                                                 unsigned int value);
//
//     void          conduit_node_set_unsigned_long(conduit_node *cnode,
//                                                  unsigned long value);
//
//    //-------------------------------------------------------------------------
//    // floating point scalar types
//    //-------------------------------------------------------------------------
//     void          conduit_node_set_float(conduit_node *cnode,
//                                          float value);
//
//     void          conduit_node_set_double(conduit_node *cnode,
//                                           double value);

//-----------------------------------------------------------------------------
// -- set via bitwidth style pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_int8_ptr(conduit_node *cnode,
                                   conduit_int8 *data,
                                   conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_int8_ptr_detailed(conduit_node *cnode,
                                            conduit_int8 *data,
                                            conduit_index_t num_elements,
                                            conduit_index_t offset,
                                            conduit_index_t stride,
                                            conduit_index_t element_bytes,
                                            conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_int16_ptr(conduit_node *cnode,
                                    conduit_int16 *data,
                                    conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_int16_ptr_detailed(conduit_node *cnode,
                                             conduit_int16 *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_int32_ptr(conduit_node *cnode,
                                    conduit_int32 *data,
                                    conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_int32_ptr_detailed(conduit_node *cnode,
                                             conduit_int32 *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_int64_ptr(conduit_node *cnode,
                                    conduit_int64 *data,
                                    conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_int64_ptr_detailed(conduit_node *cnode,
                                             conduit_int64 *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness);


    //-------------------------------------------------------------------------
    // unsigned signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_uint8_ptr(conduit_node *cnode,
                                    conduit_uint8 *data,
                                    conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_uint8_ptr_detailed(conduit_node *cnode,
                                             conduit_uint8 *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_uint16_ptr(conduit_node *cnode,
                                     conduit_uint16 *data,
                                     conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_uint16_ptr_detailed(conduit_node *cnode,
                                              conduit_uint16 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_uint32_ptr(conduit_node *cnode,
                                     conduit_uint32 *data,
                                     conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_uint32_ptr_detailed(conduit_node *cnode,
                                              conduit_uint32 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_uint64_ptr(conduit_node *cnode,
                                     conduit_uint64 *data,
                                     conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_uint64_ptr_detailed(conduit_node *cnode,
                                              conduit_uint64 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness);

    //-------------------------------------------------------------------------
    // floating point pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_float32_ptr(conduit_node *cnode,
                                      conduit_float32 *data,
                                      conduit_index_t num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_float32_ptr_detailed(conduit_node *cnode,
                                      conduit_float32 *data,
                                      conduit_index_t num_elements,
                                      conduit_index_t offset,
                                      conduit_index_t stride,
                                      conduit_index_t element_bytes,
                                      conduit_index_t endianness);
 
    //-------------------------------------------------------------------------
    void conduit_node_set_float64_ptr(conduit_node *cnode,
                                      conduit_float64 *data,
                                      conduit_index_t  num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_float64_ptr_detailed(conduit_node *cnode,
                                              conduit_float64 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness);
                                      
//-----------------------------------------------------------------------------
// -- set path for generic types --
//-----------------------------------------------------------------------------
    // void conduit_set_path_node(conduit_node *cnode,
    //                            const char* path,
    //                            conduit_node *data);

    // TODO: These req c-interfaces for datatype, schema, etc
    // //-------------------------------------------------------------------------
    // void set_path_dtype(const std::string &path,
    //                     const DataType& dtype);
    //
    // //-------------------------------------------------------------------------
    // void set_path_schema(const std::string &path,
    //                      const Schema &schema);
    //
    // //-------------------------------------------------------------------------
    // void set_path_data_using_schema(const std::string &path,
    //                                 const Schema &schema,
    //                                 void *data);
    // //-------------------------------------------------------------------------
    // void set_path_data_using_dtype(const std::string &path,
    //                                const DataType &dtype,
    //                                void *data);


//-----------------------------------------------------------------------------
// -- set path for scalar types ---
//-----------------------------------------------------------------------------
   //  //-------------------------------------------------------------------------
   //  // signed integer scalar types
   //  //-------------------------------------------------------------------------
   //  void          conduit_node_set_path_int8(conduit_node *cnode,
   //                                           const char *path,
   //                                           conduit_int8 value);
   //
   //  void          conduit_node_set_path_int16(conduit_node *cnode,
   //                                            const char *path,
   //                                            conduit_int16 value);
   //
   //  void          conduit_node_set_path_int32(conduit_node *cnode,
   //                                            const char *path,
   //                                            conduit_int32 value);
   //
   //  void          conduit_node_set_path_int64(conduit_node *cnode,
   //                                            const char *path,
   //                                            conduit_int64 value);
   // //-------------------------------------------------------------------------
   // // unsigned integer scalar types
   // //-------------------------------------------------------------------------
   //  void          conduit_node_set_path_uint8(conduit_node *cnode,
   //                                            const char *path,
   //                                            conduit_uint8 value);
   //
   //  void          conduit_node_set_path_uint16(conduit_node *cnode,
   //                                             const char *path,
   //                                             conduit_uint16 value);
   //
   //  void          conduit_node_set_path_uint32(conduit_node *cnode,
   //                                             const char *path,
   //                                             conduit_uint32 value);
   //
   //  void          conduit_node_set_path_uint64(conduit_node *cnode,
   //                                             const char *path,
   //                                             conduit_uint64 value);
   //
   // //-------------------------------------------------------------------------
   // // floating point scalar types
   // //-------------------------------------------------------------------------
   //  void          conduit_node_set_path_float32(conduit_node *cnode,
   //                                              const char *path,
   //                                              conduit_float32 value);
   //
   //  void          conduit_node_set_path_float64(conduit_node *cnode,
   //                                              const char *path,
   //                                              conduit_float64 value);


   //-------------------------------------------------------------------------
   // string case
   //-------------------------------------------------------------------------
   void          conduit_node_set_path_char8_str(conduit_node *cnode, 
                                                 const char *path,
                                                 const char *value);

//-----------------------------------------------------------------------------
// -- set path via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
//     //-------------------------------------------------------------------------
//     // signed integer pointer cases
//     //-------------------------------------------------------------------------
//     void conduit_node_set_path_int8_ptr(conduit_node *cnode,
//                                         const char *path,
//                                         conduit_int8 *data, ...);
//
//     void conduit_node_set_path_int16_ptr(conduit_node *cnode,
//                                          const char *path,
//                                          conduit_int16 *data, ...);
//
//     void conduit_node_set_path_int32_ptr(conduit_node *cnode,
//                                          const char *path,
//                                          conduit_int32 *data, ...);
//
//     void conduit_node_set_path_int64_ptr(conduit_node *cnode,
//                                          const char *path,
//                                          conduit_int64 *data, ...);
//
//     //-------------------------------------------------------------------------
//     // unsigned signed integer pointer cases
//     //-------------------------------------------------------------------------
//     void conduit_node_set_path_uint8_ptr(conduit_node *cnode,
//                                          const char *path,
//                                          conduit_uint8 *data, ...);
//
//     void conduit_node_set_path_uint16_ptr(conduit_node *cnode,
//                                           const char *path,
//                                           conduit_uint16 *data, ...);
//
//     void conduit_node_set_path_uint32_ptr(conduit_node *cnode,
//                                           const char *path,
//                                           conduit_uint32 *data, ...);
//
//     void conduit_node_set_path_uint64_ptr(conduit_node *cnode,
//                                           const char *path,
//                                           conduit_uint64 *data, ...);
//
//     //-------------------------------------------------------------------------
//     // floating point pointer cases
//     //-------------------------------------------------------------------------
//     void conduit_node_set_path_float32_ptr(conduit_node *cnode,
//                                            const char *path,
//                                            conduit_float32 *data, ...);
//
//     void conduit_node_set_path_float64_ptr(conduit_node *cnode,
//                                            const char *path,
//                                            conduit_float64 *data, ...);
//
//     //-------------------------------------------------------------------------
//     // string case
//     //-------------------------------------------------------------------------
//     void conduit_node_set_path_char8_str(conduit_node *cnode,
//                                          const char *path,
//                                          const char *value, ...);
//
//
// //-----------------------------------------------------------------------------
// // -- set external for generic types --
// //-----------------------------------------------------------------------------
//     //-------------------------------------------------------------------------
//     void conduit_node_set_external_node(conduit_node *cnode,
//                                         conduit_node *data);

    // TODO: These req c-interfaces for datatype, schema, etc
    // //-------------------------------------------------------------------------
    // void set_external_data_using_schema(const Schema &schema,
    //                                     void *data);
    //
    // //-------------------------------------------------------------------------
    // void set_external_data_using_dtype(const DataType &dtype,
    //                                    void *data);

//-----------------------------------------------------------------------------
// -- set external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
    // //-------------------------------------------------------------------------
    // // signed integer pointer cases
    // //-------------------------------------------------------------------------
    // void conduit_node_set_external_int8_ptr(conduit_node *cnode,
    //                                         conduit_int8 *data, ...);
    //
    // void conduit_node_set_external_int16_ptr(conduit_node *cnode,
    //                                          conduit_int16 *data, ...);
    //
    // void conduit_node_set_external_int32_ptr(conduit_node *cnode,
    //                                          conduit_int32 *data, ...);
    //
    // void conduit_node_set_external_int64_ptr(conduit_node *cnode,
    //                                          conduit_int64 *data, ...);
    //
    // //-------------------------------------------------------------------------
    // // unsigned signed integer pointer cases
    // //-------------------------------------------------------------------------
    // void conduit_node_set_external_uint8_ptr(conduit_node *cnode,
    //                                          conduit_uint8 *data, ...);
    //
    // void conduit_node_set_external_uint16_ptr(conduit_node *cnode,
    //                                           conduit_uint16 *data, ...);
    //
    // void conduit_node_set_external_uint32_ptr(conduit_node *cnode,
    //                                           conduit_uint32 *data, ...);
    //
    // void conduit_node_set_external_uint64_ptr(conduit_node *cnode,
    //                                           conduit_uint64 *data, ...);
    //
    // //-------------------------------------------------------------------------
    // // floating point pointer cases
    // //-------------------------------------------------------------------------
    // void conduit_node_set_external_float32_ptr(conduit_node *cnode,
    //                                            conduit_float32 *data, ...);
    //
    // void conduit_node_set_external_float64_ptr(conduit_node *cnode,
    //                                            conduit_float64 *data, ...);
    //
    // //-------------------------------------------------------------------------
    // // string case
    // //-------------------------------------------------------------------------
    // void          conduit_node_set_external_char8_str(conduit_node *cnode,
    //                                                   const char *value, ...);

//-----------------------------------------------------------------------------
// -- set path external for generic types --
//-----------------------------------------------------------------------------
    //
    // //-------------------------------------------------------------------------
    // void    set_path_external_node(conduit_node *cnode,
    //                                const char *path.
    //                                conduit_node *data);

    // TODO: These req c-interfaces for datatype, schema, etc
    // //-------------------------------------------------------------------------
    // void    set_path_external_data_using_schema(const std::string &path,
    //                                             const Schema &schema,
    //                                             void *data);
    //
    // //-------------------------------------------------------------------------
    // void    set_path_external_data_using_dtype(const std::string &path,
    //                                            const DataType &dtype,


//-----------------------------------------------------------------------------
// -- set path external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
    // //-------------------------------------------------------------------------
    // // signed integer pointer cases
    // //-------------------------------------------------------------------------
    // void conduit_node_set_path_external_int8_ptr(conduit_node *cnode,
    //                                              const char *path,
    //                                              conduit_int8 *data, ...);
    //
    // void conduit_node_set_path_external_int16_ptr(conduit_node *cnode,
    //                                               const char *path,
    //                                               conduit_int16 *data, ...);
    //
    // void conduit_node_set_path_external_int32_ptr(conduit_node *cnode,
    //                                               const char *path,
    //                                               conduit_int32 *data, ...);
    //
    // void conduit_node_set_path_external_int64_ptr(conduit_node *cnode,
    //                                               const char *path,
    //                                               conduit_int64 *data, ...);
    //
    // //-------------------------------------------------------------------------
    // // unsigned signed integer pointer cases
    // //-------------------------------------------------------------------------
    // void conduit_node_set_path_external_uint8_ptr(conduit_node *cnode,
    //                                               const char *path,
    //                                               conduit_uint8 *data, ...);
    //
    // void conduit_node_set_path_external_uint16_ptr(conduit_node *cnode,
    //                                                const char *path,
    //                                                conduit_uint16 *data, ...);
    //
    // void conduit_node_set_path_external_uint32_ptr(conduit_node *cnode,
    //                                                const char *path,
    //                                                conduit_uint32 *data, ...);
    //
    // void conduit_node_set_path_external_uint64_ptr(conduit_node *cnode,
    //                                                const char *path,
    //                                                conduit_uint64 *data, ...);
    //
    // //-------------------------------------------------------------------------
    // // floating point pointer cases
    // //-------------------------------------------------------------------------
    // void conduit_node_set_path_external_float32_ptr(conduit_node *cnode,
    //                                                 const char *path,
    //                                                 conduit_float32 *data, ...);
    //
    // void conduit_node_set_path_external_float64_ptr(conduit_node *cnode,
    //                                                 const char *path,
    //                                                 conduit_float64 *data, ...);
    //
    // //-------------------------------------------------------------------------
    // // string case
    // //-------------------------------------------------------------------------
    // void  conduit_node_set_path_external_char8_str(conduit_node *cnode,
    //                                                const char *path,
    //                                                const char *value, ...);


//-----------------------------------------------------------------------------
// Demo Interface
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void          conduit_node_set_path_int32(conduit_node *cnode,
                                          const char *path,
                                          conduit_int32 value);

//-----------------------------------------------------------------------------
void          conduit_node_set_path_float64(conduit_node *cnode,
                                            const char *path,
                                            conduit_float64 value);

//-----------------------------------------------------------------------------
void          conduit_node_set_path_char8_str(conduit_node *cnode,
                                              const char *path,
                                              const char *value);

//-----------------------------------------------------------------------------
void          conduit_node_set_int32_ptr(conduit_node *cnode,
                                         conduit_int32 *data,
                                         size_t num_elements);

//-----------------------------------------------------------------------------
void          conduit_node_set_external_int32_ptr(conduit_node *cnode,
                                                  conduit_int32 *data,
                                                  size_t num_elements);

//-----------------------------------------------------------------------------
void          conduit_node_set_path_external_float64_ptr(conduit_node *cnode,
                                                         const char *path,
                                                         conduit_float64 *data,
                                                         size_t num_elements);

//-----------------------------------------------------------------------------
void          conduit_node_set_float64(conduit_node *cnode,
                                       conduit_float64 value);

//-----------------------------------------------------------------------------
void          conduit_node_set_int(conduit_node *cnode,
                                   int value);

//-----------------------------------------------------------------------------
void          conduit_node_set_double(conduit_node *cnode,
                                      double value);

//-----------------------------------------------------------------------------
// leaf value access
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // signed integer scalar access
    //-------------------------------------------------------------------------
    conduit_int8   conduit_node_as_int8(conduit_node *cnode);
    conduit_int16  conduit_node_as_int16(conduit_node *cnode);
    conduit_int32  conduit_node_as_int32(conduit_node *cnode);
    conduit_int64  conduit_node_as_int64(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // unsigned integer scalar access
    //-------------------------------------------------------------------------
    conduit_uint8   conduit_node_as_uint8(conduit_node *cnode);
    conduit_uint16  conduit_node_as_uint16(conduit_node *cnode);
    conduit_uint32  conduit_node_as_uint32(conduit_node *cnode);
    conduit_uint64  conduit_node_as_uint64(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // floating point scalar access
    //-------------------------------------------------------------------------
    conduit_float32  conduit_node_as_float32(conduit_node *cnode);
    conduit_float64  conduit_node_as_float64(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // signed integer pointer access
    //-------------------------------------------------------------------------
    conduit_int8   *conduit_node_as_int8_ptr(conduit_node *cnode);
    conduit_int16  *conduit_node_as_int16_ptr(conduit_node *cnode);
    conduit_int32  *conduit_node_as_int32_ptr(conduit_node *cnode);
    conduit_int64  *conduit_node_as_int64_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // unsigned integer pointer access
    //-------------------------------------------------------------------------
    conduit_uint8   *conduit_node_as_uint8_ptr(conduit_node *cnode);
    conduit_uint16  *conduit_node_as_uint16_ptr(conduit_node *cnode);
    conduit_uint32  *conduit_node_as_uint32_ptr(conduit_node *cnode);
    conduit_uint64  *conduit_node_as_uint64_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // floating point pointer access
    //-------------------------------------------------------------------------
    conduit_float32  *conduit_node_as_float32_ptr(conduit_node *cnode);
    conduit_float64  *conduit_node_as_float64_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // string access
    //-------------------------------------------------------------------------
    char * conduit_node_as_char8_str(conduit_node *cnode);

//-----------------------------------------------------------------------------
// leaf value access
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
int            conduit_node_as_int(conduit_node *cnode);
//-----------------------------------------------------------------------------
int            *conduit_node_as_int_ptr(conduit_node *cnode);

//-----------------------------------------------------------------------------
conduit_int32  conduit_node_as_int32(conduit_node *cnode);
//-----------------------------------------------------------------------------
conduit_int32 *conduit_node_as_int32_ptr(conduit_node *cnode);

//-----------------------------------------------------------------------------
conduit_int32  conduit_node_fetch_path_as_int32(conduit_node *cnode,
                                                const char *path);

//-----------------------------------------------------------------------------
conduit_float64 conduit_node_fetch_path_as_float64(conduit_node *cnode,
                                                   const char *path);

//-----------------------------------------------------------------------------
char *        conduit_node_fetch_path_as_char8_str(conduit_node *cnode,
                                                   const char *path);

//-----------------------------------------------------------------------------
double        conduit_node_as_double(conduit_node *cnode);
//-----------------------------------------------------------------------------
double       *conduit_node_as_double_ptr(conduit_node *cnode);

//-----------------------------------------------------------------------------
conduit_float64 conduit_node_as_float64(conduit_node *cnode);


#ifdef __cplusplus
}
#endif

//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------


#endif
