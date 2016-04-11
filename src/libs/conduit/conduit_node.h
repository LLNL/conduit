//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
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
                                 conduit_index_t idx);

//-----------------------------------------------------------------------------
conduit_index_t conduit_node_number_of_children(conduit_node *cnode);

//-----------------------------------------------------------------------------
conduit_index_t conduit_node_number_of_elements(conduit_node *cnode);

//-----------------------------------------------------------------------------
// -- node info -- 
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
int           conduit_node_is_root(conduit_node *cnode);

//-----------------------------------------------------------------------------
void          conduit_node_print(conduit_node *cnode);
void          conduit_node_print_detailed(conduit_node *cnode);



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// Conduit Node "Set" Methods
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------


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
// -- set_path for generic types --
//-----------------------------------------------------------------------------
    void conduit_set_path_node(conduit_node *cnode,
                               const char* path,
                               conduit_node *data);

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
// -- set_external for generic types --
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    void conduit_node_set_external_node(conduit_node *cnode,
                                        conduit_node *data);

    // TODO: These req c-interfaces for datatype, schema, etc
    // //-------------------------------------------------------------------------
    // void set_external_data_using_schema(const Schema &schema,
    //                                     void *data);
    //
    // //-------------------------------------------------------------------------
    // void set_external_data_using_dtype(const DataType &dtype,
    //                                    void *data);


//-----------------------------------------------------------------------------
// -- set_path_external for generic types --
//-----------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    void    set_path_external_node(conduit_node *cnode,
                                   const char *path,
                                   conduit_node *data);

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
// -- set for string cases --
//-----------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    // set
   //-------------------------------------------------------------------------
    void    conduit_node_set_char8_str(conduit_node *cnode, 
                                       const char *value);

   //-------------------------------------------------------------------------
   // set_path
   //-------------------------------------------------------------------------
   void     conduit_node_set_path_char8_str(conduit_node *cnode, 
                                            const char *path,
                                            const char *value);

   //-------------------------------------------------------------------------
   // set_external
   //-------------------------------------------------------------------------
   void     conduit_node_set_external_char8_str(conduit_node *cnode, 
                                                char *value);

   //-------------------------------------------------------------------------
   // set_path_external
   //-------------------------------------------------------------------------
   void     conduit_node_set_path_external_char8_str(conduit_node *cnode, 
                                                     const char *path,
                                                     char *value);


//-----------------------------------------------------------------------------
// -- set for numeric types --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// For both bitwidth style and c style numeric types, we provide these variants
// of "set":
//
// * set: scalar, pointer, and pointer detailed variants
// * set_path: scalar, pointer, and pointer detailed variants
// * set_external: pointer, and pointer detailed variants 
// * set_path_external: pointer, and pointer detailed variants
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set for bitwidth style types -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for scalar bitwidth style types ---
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set bitwidth style signed integer scalar types
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
   // set bitwidth style unsigned integer scalar types
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
   // set bitwidth style floating point scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_float32(conduit_node *cnode,
                                           conduit_float32 value);

    void          conduit_node_set_float64(conduit_node *cnode,
                                           conduit_float64 value);


//-----------------------------------------------------------------------------
// -- set via bitwidth style pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set bitwidth signed integer pointer cases
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
    // set bitwidth unsigned signed integer pointer cases
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
    // set bitwidth floating point pointer cases
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
//-----------------------------------------------------------------------------
// -- set_path for bitwidth style types --
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set path for bitwidth scalar types ---
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set_path bitwidth signed integer scalar types
    //-------------------------------------------------------------------------
    void          conduit_node_set_path_int8(conduit_node *cnode,
                                             const char *path,
                                             conduit_int8 value);

    void          conduit_node_set_path_int16(conduit_node *cnode,
                                              const char *path,
                                              conduit_int16 value);

    void          conduit_node_set_path_int32(conduit_node *cnode,
                                              const char *path,
                                              conduit_int32 value);

    void          conduit_node_set_path_int64(conduit_node *cnode,
                                              const char *path,
                                              conduit_int64 value);
   //-------------------------------------------------------------------------
   // set_path bitwidth unsigned integer scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_path_uint8(conduit_node *cnode,
                                              const char *path,
                                              conduit_uint8 value);

    void          conduit_node_set_path_uint16(conduit_node *cnode,
                                               const char *path,
                                               conduit_uint16 value);

    void          conduit_node_set_path_uint32(conduit_node *cnode,
                                               const char *path,
                                               conduit_uint32 value);

    void          conduit_node_set_path_uint64(conduit_node *cnode,
                                               const char *path,
                                               conduit_uint64 value);

   //-------------------------------------------------------------------------
   // set_path floating point scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_path_float32(conduit_node *cnode,
                                                const char *path,
                                                conduit_float32 value);

    void          conduit_node_set_path_float64(conduit_node *cnode,
                                                const char *path,
                                                conduit_float64 value);


//-----------------------------------------------------------------------------
// -- set path via bitwidth style pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set_path bitwidth signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_int8_ptr(conduit_node *cnode,
                                        const char *path,
                                        conduit_int8 *data,
                                        conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int8_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 conduit_int8 *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int16_ptr(conduit_node *cnode,
                                        const char *path,
                                        conduit_int16 *data,
                                        conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int16_ptr_detailed(conduit_node *cnode,
                                                  const char *path,
                                                  conduit_int16 *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int32_ptr(conduit_node *cnode,
                                         const char *path,
                                         conduit_int32 *data,
                                         conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int32_ptr_detailed(conduit_node *cnode,
                                                  const char *path,
                                                  conduit_int32 *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int64_ptr(conduit_node *cnode,
                                         const char *path,
                                         conduit_int64 *data,
                                         conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int64_ptr_detailed(conduit_node *cnode,
                                                  const char *path,
                                                  conduit_int64 *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness);


    //-------------------------------------------------------------------------
    // set_path bitwidth unsigned signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint8_ptr(conduit_node *cnode,
                                         const char *path,
                                         conduit_uint8 *data,
                                         conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint8_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 conduit_uint8 *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint16_ptr(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint16 *data,
                                          conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint16_ptr_detailed(conduit_node *cnode,
                                                   const char *path,
                                                   conduit_uint16 *data,
                                                   conduit_index_t num_elements,
                                                   conduit_index_t offset,
                                                   conduit_index_t stride,
                                                   conduit_index_t element_bytes,
                                                   conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint32_ptr(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint32 *data,
                                          conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint32_ptr_detailed(conduit_node *cnode,
                                                   const char *path,
                                                   conduit_uint32 *data,
                                                   conduit_index_t num_elements,
                                                   conduit_index_t offset,
                                                   conduit_index_t stride,
                                                   conduit_index_t element_bytes,
                                                   conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint64_ptr(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint64 *data,
                                          conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_uint64_ptr_detailed(conduit_node *cnode,
                                                   const char *path,
                                                   conduit_uint64 *data,
                                                   conduit_index_t num_elements,
                                                   conduit_index_t offset,
                                                   conduit_index_t stride,
                                                   conduit_index_t element_bytes,
                                                   conduit_index_t endianness);

    //-------------------------------------------------------------------------
    // set_path bitwidth floating point pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_float32_ptr(conduit_node *cnode,
                                           const char *path,
                                           conduit_float32 *data,
                                           conduit_index_t num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_path_float32_ptr_detailed(conduit_node *cnode,
                                                    const char *path,
                                                    conduit_float32 *data,
                                                    conduit_index_t num_elements,
                                                    conduit_index_t offset,
                                                    conduit_index_t stride,
                                                    conduit_index_t element_bytes,
                                                    conduit_index_t endianness);
 
    //-------------------------------------------------------------------------
    void conduit_node_set_path_float64_ptr(conduit_node *cnode,
                                           const char *path,
                                           conduit_float64 *data,
                                           conduit_index_t  num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_path_float64_ptr_detailed(conduit_node *cnode,
                                                    const char *path,
                                                    conduit_float64 *data,
                                                    conduit_index_t num_elements,
                                                    conduit_index_t offset,
                                                    conduit_index_t stride,
                                                    conduit_index_t element_bytes,
                                                    conduit_index_t endianness);


//-----------------------------------------------------------------------------
// -- set_external via pointers (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set_external bitwidth signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_external_int8_ptr(conduit_node *cnode,
                                            conduit_int8 *data,
                                            conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int8_ptr_detailed(conduit_node *cnode,
                                                     conduit_int8 *data,
                                                     conduit_index_t num_elements,
                                                     conduit_index_t offset,
                                                     conduit_index_t stride,
                                                     conduit_index_t element_bytes,
                                                     conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int16_ptr(conduit_node *cnode,
                                             conduit_int16 *data,
                                             conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int16_ptr_detailed(conduit_node *cnode,
                                                      conduit_int16 *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int32_ptr(conduit_node *cnode,
                                             conduit_int32 *data,
                                             conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int32_ptr_detailed(conduit_node *cnode,
                                                      conduit_int32 *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int64_ptr(conduit_node *cnode,
                                             conduit_int64 *data,
                                             conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int64_ptr_detailed(conduit_node *cnode,
                                                      conduit_int64 *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness);


    //-------------------------------------------------------------------------
    // set_external bitwidth unsigned signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint8_ptr(conduit_node *cnode,
                                             conduit_uint8 *data,
                                             conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint8_ptr_detailed(conduit_node *cnode,
                                                      conduit_uint8 *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint16_ptr(conduit_node *cnode,
                                              conduit_uint16 *data,
                                              conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint16_ptr_detailed(conduit_node *cnode,
                                                       conduit_uint16 *data,
                                                       conduit_index_t num_elements,
                                                       conduit_index_t offset,
                                                       conduit_index_t stride,
                                                       conduit_index_t element_bytes,
                                                       conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint32_ptr(conduit_node *cnode,
                                              conduit_uint32 *data,
                                              conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint32_ptr_detailed(conduit_node *cnode,
                                                       conduit_uint32 *data,
                                                       conduit_index_t num_elements,
                                                       conduit_index_t offset,
                                                       conduit_index_t stride,
                                                       conduit_index_t element_bytes,
                                                       conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint64_ptr(conduit_node *cnode,
                                              conduit_uint64 *data,
                                              conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_uint64_ptr_detailed(conduit_node *cnode,
                                                       conduit_uint64 *data,
                                                       conduit_index_t num_elements,
                                                       conduit_index_t offset,
                                                       conduit_index_t stride,   
                                                       conduit_index_t element_bytes,
                                                       conduit_index_t endianness);

    //-------------------------------------------------------------------------
    // set_external bitwidth floating point pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_external_float32_ptr(conduit_node *cnode,
                                               conduit_float32 *data,
                                               conduit_index_t num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_external_float32_ptr_detailed(conduit_node *cnode,
                                                        conduit_float32 *data,
                                                        conduit_index_t num_elements,
                                                        conduit_index_t offset,
                                                        conduit_index_t stride,
                                                        conduit_index_t element_bytes,
                                                        conduit_index_t endianness);
 
    //-------------------------------------------------------------------------
    void conduit_node_set_external_float64_ptr(conduit_node *cnode,
                                               conduit_float64 *data,
                                               conduit_index_t  num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_external_float64_ptr_detailed(conduit_node *cnode,
                                                        conduit_float64 *data,
                                                        conduit_index_t num_elements,
                                                        conduit_index_t offset,
                                                        conduit_index_t stride,
                                                        conduit_index_t element_bytes,
                                                        conduit_index_t endianness);


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set for cstyle types -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for scalar cstyle types ---
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set cstyle signed integer scalar types
    //-------------------------------------------------------------------------
    void          conduit_node_set_char(conduit_node *cnode,
                                        char value);
                                     
    void          conduit_node_set_short(conduit_node *cnode,
                                         short value);

    void          conduit_node_set_int(conduit_node *cnode,
                                       int value);

    void          conduit_node_set_long(conduit_node *cnode,
                                        long value);
   //-------------------------------------------------------------------------
   // set cstyle unsigned integer scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_unsigned_char(conduit_node *cnode,
                                                 unsigned char value);

    void          conduit_node_set_unsigned_short(conduit_node *cnode,
                                                  unsigned short value);

    void          conduit_node_set_unsigned_int(conduit_node *cnode,
                                                unsigned int value);

    void          conduit_node_set_unsigned_long(conduit_node *cnode,
                                                 unsigned long value);

   //-------------------------------------------------------------------------
   // set cstyle floating point scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_float(conduit_node *cnode,
                                         float value);

    void          conduit_node_set_double(conduit_node *cnode,
                                          double value);

//-----------------------------------------------------------------------------
// -- set via cstyle pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set cstyle signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_char_ptr(conduit_node *cnode,
                                   char *data,
                                   conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_char_ptr_detailed(conduit_node *cnode,
                                            char *data,
                                            conduit_index_t num_elements,
                                            conduit_index_t offset,
                                            conduit_index_t stride,
                                            conduit_index_t element_bytes,
                                            conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_short_ptr(conduit_node *cnode,
                                    short *data,
                                    conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_short_ptr_detailed(conduit_node *cnode,
                                             short *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_int_ptr(conduit_node *cnode,
                                  int *data,
                                  conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_int_ptr_detailed(conduit_node *cnode,
                                           int *data,
                                           conduit_index_t num_elements,
                                           conduit_index_t offset,
                                           conduit_index_t stride,
                                           conduit_index_t element_bytes,
                                           conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_long_ptr(conduit_node *cnode,
                                   long *data,
                                    conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_long_ptr_detailed(conduit_node *cnode,
                                            long *data,
                                            conduit_index_t num_elements,
                                            conduit_index_t offset,
                                            conduit_index_t stride,
                                            conduit_index_t element_bytes,
                                            conduit_index_t endianness);


    //-------------------------------------------------------------------------
    // set cstyle unsigned integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_char_ptr(conduit_node *cnode,
                                            unsigned char *data,
                                            conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_char_ptr_detailed(conduit_node *cnode,
                                                     unsigned char *data,
                                                     conduit_index_t num_elements,
                                                     conduit_index_t offset,
                                                     conduit_index_t stride,
                                                     conduit_index_t element_bytes,
                                                     conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_short_ptr(conduit_node *cnode,
                                             unsigned short *data,
                                             conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_short_ptr_detailed(conduit_node *cnode,
                                                      unsigned short *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_int_ptr(conduit_node *cnode,
                                           unsigned int *data,
                                           conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_int_ptr_detailed(conduit_node *cnode,
                                                    unsigned int *data,
                                                    conduit_index_t num_elements,
                                                    conduit_index_t offset,
                                                    conduit_index_t stride,
                                                    conduit_index_t element_bytes,
                                                    conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_long_ptr(conduit_node *cnode,
                                            unsigned long *data,
                                            conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_unsigned_long_ptr_detailed(conduit_node *cnode,
                                                     unsigned long *data,
                                                     conduit_index_t num_elements,
                                                     conduit_index_t offset,
                                                     conduit_index_t stride,
                                                     conduit_index_t element_bytes,
                                                     conduit_index_t endianness);

    //-------------------------------------------------------------------------
    // set cstyle floating point pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_float_ptr(conduit_node *cnode,
                                    float *data,
                                    conduit_index_t num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_float_ptr_detailed(conduit_node *cnode,
                                             float *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness);
 
    //-------------------------------------------------------------------------
    void conduit_node_set_double_ptr(conduit_node *cnode,
                                     double *data,
                                     conduit_index_t  num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_double_ptr_detailed(conduit_node *cnode,
                                              double *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness);

//-----------------------------------------------------------------------------
// -- set_path for scalar cstyle types ---
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set_path cstyle signed integer scalar types
    //-------------------------------------------------------------------------
    void          conduit_node_set_path_char(conduit_node *cnode,
                                             const char *path,
                                             char value);
                                     
    void          conduit_node_set_path_short(conduit_node *cnode,
                                              const char *path,
                                              short value);

    void          conduit_node_set_path_int(conduit_node *cnode,
                                            const char *path,
                                            int value);

    void          conduit_node_set_path_long(conduit_node *cnode,
                                             const char *path,
                                             long value);
   //-------------------------------------------------------------------------
   // set_path cstyle unsigned integer scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_path_unsigned_char(conduit_node *cnode,
                                                      const char *path,
                                                      unsigned char value);

    void          conduit_node_set_path_unsigned_short(conduit_node *cnode,
                                                       const char *path,
                                                       unsigned short value);

    void          conduit_node_set_path_unsigned_int(conduit_node *cnode,
                                                     const char *path,
                                                     unsigned int value);

    void          conduit_node_set_path_unsigned_long(conduit_node *cnode,
                                                      const char *path,
                                                      unsigned long value);

   //-------------------------------------------------------------------------
   // set_path cstyle floating point scalar types
   //-------------------------------------------------------------------------
    void          conduit_node_set_path_float(conduit_node *cnode,
                                              const char *path,
                                              float value);

    void          conduit_node_set_path_double(conduit_node *cnode,
                                               const char *path,
                                               double value);

//-----------------------------------------------------------------------------
// -- set_path via cstyle pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set_path cstyle signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_char_ptr(conduit_node *cnode,
                                        const char *path,
                                        char *data,
                                        conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_char_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 char *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_short_ptr(conduit_node *cnode,
                                         const char *path,
                                         short *data,
                                         conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_short_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 short *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int_ptr(conduit_node *cnode,
                                       const char *path,
                                       int *data,
                                       conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_int_ptr_detailed(conduit_node *cnode,
                                                const char *path,
                                                int *data,
                                                conduit_index_t num_elements,
                                                conduit_index_t offset,
                                                conduit_index_t stride,
                                                conduit_index_t element_bytes,
                                                conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_long_ptr(conduit_node *cnode,
                                        const char *path,
                                        long *data,
                                        conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_long_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 long *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness);


    //-------------------------------------------------------------------------
    // set_path cstyle unsigned integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_char_ptr(conduit_node *cnode,
                                                 const char *path,
                                                 unsigned char *data,
                                                 conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_char_ptr_detailed(conduit_node *cnode,
                                                          const char *path,
                                                          unsigned char *data,
                                                          conduit_index_t num_elements,
                                                          conduit_index_t offset,
                                                          conduit_index_t stride,
                                                          conduit_index_t element_bytes,
                                                          conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_short_ptr(conduit_node *cnode,
                                                  const char *path,
                                                  unsigned short *data,
                                                  conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_short_ptr_detailed(conduit_node *cnode,
                                                           const char *path,
                                                           unsigned short *data,
                                                           conduit_index_t num_elements,
                                                           conduit_index_t offset,
                                                           conduit_index_t stride,
                                                           conduit_index_t element_bytes,
                                                           conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_int_ptr(conduit_node *cnode,
                                                const char *path,
                                                unsigned int *data,
                                                conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_int_ptr_detailed(conduit_node *cnode,
                                                         const char *path,
                                                         unsigned int *data,
                                                         conduit_index_t num_elements,
                                                         conduit_index_t offset,
                                                         conduit_index_t stride,
                                                         conduit_index_t element_bytes,
                                                         conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_long_ptr(conduit_node *cnode,
                                                 const char *path,
                                                 unsigned long *data,
                                                 conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_unsigned_long_ptr_detailed(conduit_node *cnode,
                                                          const char *path,
                                                          unsigned long *data,
                                                          conduit_index_t num_elements,
                                                          conduit_index_t offset,
                                                          conduit_index_t stride,
                                                          conduit_index_t element_bytes,
                                                          conduit_index_t endianness);

    //-------------------------------------------------------------------------
    // set_path cstyle floating point pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_float_ptr(conduit_node *cnode,
                                         const char *path,
                                         float *data,
                                         conduit_index_t num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_path_float_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 float *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness);
 
    //-------------------------------------------------------------------------
    void conduit_node_set_path_double_ptr(conduit_node *cnode,
                                          const char *path,
                                          double *data,
                                          conduit_index_t  num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_path_double_ptr_detailed(conduit_node *cnode,
                                                   const char *path,
                                                   double *data,
                                                   conduit_index_t num_elements,
                                                   conduit_index_t offset,
                                                   conduit_index_t stride,
                                                   conduit_index_t element_bytes,
                                                   conduit_index_t endianness);

//-----------------------------------------------------------------------------
// -- set_external via cstyle pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set_external cstyle signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_external_char_ptr(conduit_node *cnode,
                                            char *data,
                                            conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_char_ptr_detailed(conduit_node *cnode,
                                                     char *data,
                                                     conduit_index_t num_elements,
                                                     conduit_index_t offset,
                                                     conduit_index_t stride,
                                                     conduit_index_t element_bytes,
                                                     conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_short_ptr(conduit_node *cnode,
                                             short *data,
                                             conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_short_ptr_detailed(conduit_node *cnode,
                                                      short *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int_ptr(conduit_node *cnode,
                                           int *data,
                                           conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_int_ptr_detailed(conduit_node *cnode,
                                                    int *data,
                                                    conduit_index_t num_elements,
                                                    conduit_index_t offset,
                                                    conduit_index_t stride,
                                                    conduit_index_t element_bytes,
                                                    conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_long_ptr(conduit_node *cnode,
                                            long *data,
                                            conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_long_ptr_detailed(conduit_node *cnode,
                                                     long *data,
                                                     conduit_index_t num_elements,
                                                     conduit_index_t offset,
                                                     conduit_index_t stride,
                                                     conduit_index_t element_bytes,
                                                     conduit_index_t endianness);


    //-------------------------------------------------------------------------
    // set_external cstyle unsigned integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_char_ptr(conduit_node *cnode,
                                                     unsigned char *data,
                                                     conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_char_ptr_detailed(conduit_node *cnode,
                                                              unsigned char *data,
                                                              conduit_index_t num_elements,
                                                              conduit_index_t offset,
                                                              conduit_index_t stride,
                                                              conduit_index_t element_bytes,
                                                              conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_short_ptr(conduit_node *cnode,
                                                      unsigned short *data,
                                                      conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_short_ptr_detailed(conduit_node *cnode,
                                                               unsigned short *data,
                                                               conduit_index_t num_elements,
                                                               conduit_index_t offset,
                                                               conduit_index_t stride,
                                                               conduit_index_t element_bytes,
                                                               conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_int_ptr(conduit_node *cnode,
                                                    unsigned int *data,
                                                    conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_int_ptr_detailed(conduit_node *cnode,
                                                             unsigned int *data,
                                                             conduit_index_t num_elements,
                                                             conduit_index_t offset,
                                                             conduit_index_t stride,
                                                             conduit_index_t element_bytes,
                                                             conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_long_ptr(conduit_node *cnode,
                                                     unsigned long *data,
                                                     conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_external_unsigned_long_ptr_detailed(conduit_node *cnode,
                                                              unsigned long *data,
                                                              conduit_index_t num_elements,
                                                              conduit_index_t offset,
                                                              conduit_index_t stride,
                                                              conduit_index_t element_bytes,
                                                              conduit_index_t endianness);

    //-------------------------------------------------------------------------
    // set_external cstyle floating point pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_external_float_ptr(conduit_node *cnode,
                                             float *data,
                                             conduit_index_t num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_external_float_ptr_detailed(conduit_node *cnode,
                                                      float *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness);
 
    //-------------------------------------------------------------------------
    void conduit_node_set_external_double_ptr(conduit_node *cnode,
                                              double *data,
                                              conduit_index_t  num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_external_double_ptr_detailed(conduit_node *cnode,
                                                       double *data,
                                                       conduit_index_t num_elements,
                                                       conduit_index_t offset,
                                                       conduit_index_t stride,
                                                       conduit_index_t element_bytes,
                                                       conduit_index_t endianness);

//-----------------------------------------------------------------------------
// -- set_path_external via cstyle pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // set_path_external cstyle signed integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_char_ptr(conduit_node *cnode,
                                                 const char *path,
                                                 char *data,
                                                 conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_char_ptr_detailed(conduit_node *cnode,
                                                          const char *path,
                                                          char *data,
                                                          conduit_index_t num_elements,
                                                          conduit_index_t offset,
                                                          conduit_index_t stride,
                                                          conduit_index_t element_bytes,
                                                          conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_short_ptr(conduit_node *cnode,
                                                  const char *path,
                                                  short *data,
                                                  conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_short_ptr_detailed(conduit_node *cnode,
                                                           const char *path,
                                                           short *data,
                                                           conduit_index_t num_elements,
                                                           conduit_index_t offset,
                                                           conduit_index_t stride,
                                                           conduit_index_t element_bytes,
                                                           conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_int_ptr(conduit_node *cnode,
                                                const char *path,
                                                int *data,
                                                conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_int_ptr_detailed(conduit_node *cnode,
                                                         const char *path,
                                                         int *data,
                                                         conduit_index_t num_elements,
                                                         conduit_index_t offset,
                                                         conduit_index_t stride,
                                                         conduit_index_t element_bytes,
                                                         conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_long_ptr(conduit_node *cnode,
                                                 const char *path,
                                                 long *data,
                                                 conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_long_ptr_detailed(conduit_node *cnode,
                                                          const char *path,
                                                          long *data,
                                                          conduit_index_t num_elements,
                                                          conduit_index_t offset,
                                                          conduit_index_t stride,
                                                          conduit_index_t element_bytes,
                                                          conduit_index_t endianness);


    //-------------------------------------------------------------------------
    // set_path cstyle unsigned integer pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_char_ptr(conduit_node *cnode,
                                                          const char *path,
                                                          unsigned char *data,
                                                          conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_char_ptr_detailed(conduit_node *cnode,
                                                                   const char *path,
                                                                   unsigned char *data,
                                                                   conduit_index_t num_elements,
                                                                   conduit_index_t offset,
                                                                   conduit_index_t stride,
                                                                   conduit_index_t element_bytes,
                                                                   conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_short_ptr(conduit_node *cnode,
                                                           const char *path,
                                                           unsigned short *data,
                                                           conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_short_ptr_detailed(conduit_node *cnode,
                                                                    const char *path,
                                                                    unsigned short *data,
                                                                    conduit_index_t num_elements,
                                                                    conduit_index_t offset,
                                                                    conduit_index_t stride,
                                                                    conduit_index_t element_bytes,
                                                                    conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_int_ptr(conduit_node *cnode,
                                                         const char *path,
                                                         unsigned int *data,
                                                         conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_int_ptr_detailed(conduit_node *cnode,
                                                                  const char *path,
                                                                  unsigned int *data,
                                                                  conduit_index_t num_elements,
                                                                  conduit_index_t offset,
                                                                  conduit_index_t stride,
                                                                  conduit_index_t element_bytes,
                                                                  conduit_index_t endianness);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_long_ptr(conduit_node *cnode,
                                                          const char *path,
                                                          unsigned long *data,
                                                          conduit_index_t num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_unsigned_long_ptr_detailed(conduit_node *cnode,
                                                                   const char *path,
                                                                   unsigned long *data,
                                                                   conduit_index_t num_elements,
                                                                   conduit_index_t offset,
                                                                   conduit_index_t stride,
                                                                   conduit_index_t element_bytes,
                                                                   conduit_index_t endianness);

    //-------------------------------------------------------------------------
    // set_path cstyle floating point pointer cases
    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_float_ptr(conduit_node *cnode,
                                                  const char *path,
                                                  float *data,
                                                  conduit_index_t num_elements);
    
    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_float_ptr_detailed(conduit_node *cnode,
                                                           const char *path,
                                                           float *data,
                                                           conduit_index_t num_elements,
                                                           conduit_index_t offset,
                                                           conduit_index_t stride,
                                                           conduit_index_t element_bytes,
                                                           conduit_index_t endianness);
 
    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_double_ptr(conduit_node *cnode,
                                                   const char *path,
                                                   double *data,
                                                   conduit_index_t  num_elements);

    //-------------------------------------------------------------------------
    void conduit_node_set_path_external_double_ptr_detailed(conduit_node *cnode,
                                                            const char *path,
                                                            double *data,
                                                            conduit_index_t num_elements,
                                                            conduit_index_t offset,
                                                            conduit_index_t stride,
                                                            conduit_index_t element_bytes,
                                                            conduit_index_t endianness);


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// leaf access
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// leaf value access (generic)
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // direct data access
    //-------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    // direct data pointer access 
    //-------------------------------------------------------------------------
    void *conduit_node_data_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // element pointer access 
    //-------------------------------------------------------------------------
    void *conduit_node_element_ptr(conduit_node *cnode,
                                   conduit_index_t idx);

    //-------------------------------------------------------------------------
    // fetch_path direct data pointer access 
    //-------------------------------------------------------------------------
    void *conduit_node_fetch_path_data_ptr(conduit_node *cnode,
                                           const char* path);

    //-------------------------------------------------------------------------
    // fetch_path element pointer access 
    //-------------------------------------------------------------------------
    void *conduit_node_fetch_path_element_ptr(conduit_node *cnode,
                                              const char* path,
                                              conduit_index_t idx);

    //-------------------------------------------------------------------------
    // string access
    //-------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    // as string
    //-------------------------------------------------------------------------
    char *conduit_node_as_char8_str(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // fetch_path_as string
    //-------------------------------------------------------------------------
    char *conduit_node_fetch_path_as_char8_str(conduit_node *cnode,
                                               const char *path);

//-----------------------------------------------------------------------------
// leaf value access (bitwidth style types)
//-----------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    // as bitwidth signed integer scalar access
    //-------------------------------------------------------------------------
    conduit_int8   conduit_node_as_int8(conduit_node *cnode);
    conduit_int16  conduit_node_as_int16(conduit_node *cnode);
    conduit_int32  conduit_node_as_int32(conduit_node *cnode);
    conduit_int64  conduit_node_as_int64(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as bitwidth unsigned integer scalar access
    //-------------------------------------------------------------------------
    conduit_uint8   conduit_node_as_uint8(conduit_node *cnode);
    conduit_uint16  conduit_node_as_uint16(conduit_node *cnode);
    conduit_uint32  conduit_node_as_uint32(conduit_node *cnode);
    conduit_uint64  conduit_node_as_uint64(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as bitwidth floating point scalar access
    //-------------------------------------------------------------------------
    conduit_float32  conduit_node_as_float32(conduit_node *cnode);
    conduit_float64  conduit_node_as_float64(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as bitwidth signed integer pointer access
    //-------------------------------------------------------------------------
    conduit_int8   *conduit_node_as_int8_ptr(conduit_node *cnode);
    conduit_int16  *conduit_node_as_int16_ptr(conduit_node *cnode);
    conduit_int32  *conduit_node_as_int32_ptr(conduit_node *cnode);
    conduit_int64  *conduit_node_as_int64_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as bitwidth unsigned integer pointer access
    //-------------------------------------------------------------------------
    conduit_uint8   *conduit_node_as_uint8_ptr(conduit_node *cnode);
    conduit_uint16  *conduit_node_as_uint16_ptr(conduit_node *cnode);
    conduit_uint32  *conduit_node_as_uint32_ptr(conduit_node *cnode);
    conduit_uint64  *conduit_node_as_uint64_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as bitwidth floating point pointer access
    //-------------------------------------------------------------------------
    conduit_float32  *conduit_node_as_float32_ptr(conduit_node *cnode);
    conduit_float64  *conduit_node_as_float64_ptr(conduit_node *cnode);


//-----------------------------------------------------------------------------
// leaf value access via path (bitwidth style types)
//-----------------------------------------------------------------------------

    //-------------------------------------------------------------------------
    // fetch_path_as bitwidth signed integer scalar access
    //-------------------------------------------------------------------------
    conduit_int8   conduit_node_fetch_path_as_int8(conduit_node *cnode,
                                                   const char *path);
    conduit_int16  conduit_node_fetch_path_as_int16(conduit_node *cnode,
                                                    const char *path);
    conduit_int32  conduit_node_fetch_path_as_int32(conduit_node *cnode,
                                                    const char *path);
    conduit_int64  conduit_node_fetch_path_as_int64(conduit_node *cnode,
                                                    const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as bitwidth unsigned integer scalar access
    //-------------------------------------------------------------------------
    conduit_uint8   conduit_node_fetch_path_as_uint8(conduit_node *cnode,
                                                     const char *path);
    conduit_uint16  conduit_node_fetch_path_as_uint16(conduit_node *cnode,
                                                      const char *path);
    conduit_uint32  conduit_node_fetch_path_as_uint32(conduit_node *cnode,
                                                      const char *path);
    conduit_uint64  conduit_node_fetch_path_as_uint64(conduit_node *cnode,
                                                      const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as bitwidth floating point scalar access
    //-------------------------------------------------------------------------
    conduit_float32  conduit_node_fetch_path_as_float32(conduit_node *cnode,
                                                        const char *path);
    conduit_float64  conduit_node_fetch_path_as_float64(conduit_node *cnode,
                                                        const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as bitwidth signed integer pointer access
    //-------------------------------------------------------------------------
    conduit_int8   *conduit_node_fetch_path_as_int8_ptr(conduit_node *cnode,
                                                        const char *path);
    conduit_int16  *conduit_node_fetch_path_as_int16_ptr(conduit_node *cnode,
                                                         const char *path);
    conduit_int32  *conduit_node_fetch_path_as_int32_ptr(conduit_node *cnode,
                                                         const char *path);
    conduit_int64  *conduit_node_fetch_path_as_int64_ptr(conduit_node *cnode,
                                                         const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as bitwidth unsigned integer pointer access
    //-------------------------------------------------------------------------
    conduit_uint8   *conduit_node_fetch_path_as_uint8_ptr(conduit_node *cnode,
                                                          const char *path);
    conduit_uint16  *conduit_node_fetch_path_as_uint16_ptr(conduit_node *cnode,
                                                           const char *path);
    conduit_uint32  *conduit_node_fetch_path_as_uint32_ptr(conduit_node *cnode,
                                                           const char *path);
    conduit_uint64  *conduit_node_fetch_path_as_uint64_ptr(conduit_node *cnode,
                                                           const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as bitwidth floating point pointer access
    //-------------------------------------------------------------------------
    conduit_float32  *conduit_node_fetch_path_as_float32_ptr(conduit_node *cnode,
                                                             const char *path);
    conduit_float64  *conduit_node_fetch_path_as_float64_ptr(conduit_node *cnode,
                                                             const char *path);


//-----------------------------------------------------------------------------
// leaf value access (native c style types)
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // as cstyle signed integer scalar access
    //-------------------------------------------------------------------------
    char  conduit_node_as_char(conduit_node *cnode);
    short conduit_node_as_short(conduit_node *cnode);
    int   conduit_node_as_int(conduit_node *cnode);
    long  conduit_node_as_long(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as cstyle unsigned integer scalar access
    //-------------------------------------------------------------------------
    unsigned char   conduit_node_as_unsigned_char(conduit_node *cnode);
    unsigned short  conduit_node_as_unsigned_short(conduit_node *cnode);
    unsigned int    conduit_node_as_unsigned_int(conduit_node *cnode);
    unsigned long   conduit_node_as_unsigned_long(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as cstyle floating point scalar access
    //-------------------------------------------------------------------------
    float  conduit_node_as_float(conduit_node *cnode);
    double conduit_node_as_double(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as cstyle signed integer pointer access
    //-------------------------------------------------------------------------
    char   *conduit_node_as_char_ptr(conduit_node *cnode);
    short  *conduit_node_as_short_ptr(conduit_node *cnode);
    int    *conduit_node_as_int_ptr(conduit_node *cnode);
    long   *conduit_node_as_long_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as cstyle unsigned integer pointer access
    //-------------------------------------------------------------------------
    unsigned char   *conduit_node_as_unsigned_char_ptr(conduit_node *cnode);
    unsigned short  *conduit_node_as_unsigned_short_ptr(conduit_node *cnode);
    unsigned int    *conduit_node_as_unsigned_int_ptr(conduit_node *cnode);
    unsigned long   *conduit_node_as_unsigned_long_ptr(conduit_node *cnode);

    //-------------------------------------------------------------------------
    // as cstyle floating point pointer access
    //-------------------------------------------------------------------------
    float   *conduit_node_as_float_ptr(conduit_node *cnode);
    double  *conduit_node_as_double_ptr(conduit_node *cnode);

//-----------------------------------------------------------------------------
// leaf value access via path (native c style types)
//-----------------------------------------------------------------------------
    //-------------------------------------------------------------------------
    // fetch_path_as cstyle signed integer scalar access
    //-------------------------------------------------------------------------
    char  conduit_node_fetch_path_as_char(conduit_node *cnode,
                                          const char *path);
    short conduit_node_fetch_path_as_short(conduit_node *cnode,
                                          const char *path);
    int   conduit_node_fetch_path_as_int(conduit_node *cnode,
                                          const char *path);
    long  conduit_node_fetch_path_as_long(conduit_node *cnode,
                                          const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as cstyle unsigned integer scalar access
    //-------------------------------------------------------------------------
    unsigned char   conduit_node_fetch_path_as_unsigned_char(conduit_node *cnode,
                                                             const char *path);
    unsigned short  conduit_node_fetch_path_as_unsigned_short(conduit_node *cnode,
                                                              const char *path);
    unsigned int    conduit_node_fetch_path_as_unsigned_int(conduit_node *cnode,
                                                            const char *path);
    unsigned long   conduit_node_fetch_path_as_unsigned_long(conduit_node *cnode,
                                                             const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as cstyle floating point scalar access
    //-------------------------------------------------------------------------
    float  conduit_node_fetch_path_as_float(conduit_node *cnode,
                                            const char *path);
    double conduit_node_fetch_path_as_double(conduit_node *cnode,
                                             const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as cstyle signed integer pointer access
    //-------------------------------------------------------------------------
    char   *conduit_node_fetch_path_as_char_ptr(conduit_node *cnode,
                                                 const char *path);
    short  *conduit_node_fetch_path_as_short_ptr(conduit_node *cnode,
                                                 const char *path);
    int    *conduit_node_fetch_path_as_int_ptr(conduit_node *cnode,
                                               const char *path);
    long   *conduit_node_fetch_path_as_long_ptr(conduit_node *cnode,
                                                const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as cstyle unsigned integer pointer access
    //-------------------------------------------------------------------------
    unsigned char   *conduit_node_fetch_path_as_unsigned_char_ptr(conduit_node *cnode,
                                                                  const char *path);
    unsigned short  *conduit_node_fetch_path_as_unsigned_short_ptr(conduit_node *cnode,
                                                                   const char *path);
    unsigned int    *conduit_node_fetch_path_as_unsigned_int_ptr(conduit_node *cnode,
                                                                  const char *path);
    unsigned long   *conduit_node_fetch_path_as_unsigned_long_ptr(conduit_node *cnode,
                                                                  const char *path);

    //-------------------------------------------------------------------------
    // fetch_path_as cstyle floating point pointer access
    //-------------------------------------------------------------------------
    float   *conduit_node_fetch_path_as_float_ptr(conduit_node *cnode,
                                                  const char *path);
    double  *conduit_node_fetch_path_as_double_ptr(conduit_node *cnode,
                                                  const char *path);

#ifdef __cplusplus
}
#endif

//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------


#endif
