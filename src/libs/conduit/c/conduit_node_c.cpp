//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: conduit_node_c.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_node.h"

#include "conduit.hpp"
#include "conduit_cpp_to_c.hpp"

#include <stdlib.h>

//-----------------------------------------------------------------------------
// -- begin extern C
//-----------------------------------------------------------------------------

extern "C" {

using namespace conduit;

//-----------------------------------------------------------------------------
// -- basic constructor and destruction -- 
//-----------------------------------------------------------------------------


//---------------------------------------------------------------------------//
conduit_node *
conduit_node_create()
{
    return c_node(new Node());
}

//---------------------------------------------------------------------------//
void
conduit_node_destroy(conduit_node *cnode)
{
    Node *n = cpp_node(cnode);
    // only clean up if n is the root node (not owned by another node)
    if(n->is_root())
    {
        delete n;
    }
}

//-----------------------------------------------------------------------------
conduit_node *
conduit_node_fetch(conduit_node *cnode,
                   const char *path)
{
    return c_node(cpp_node(cnode)->fetch_ptr(path));
}

//-----------------------------------------------------------------------------
conduit_node *
conduit_node_append(conduit_node *cnode)
{
    return c_node(&cpp_node(cnode)->append());
}

//-----------------------------------------------------------------------------
conduit_node *
conduit_node_child(conduit_node *cnode, conduit_index_t idx)
{
    return c_node(cpp_node(cnode)->child_ptr(idx));
}

//-----------------------------------------------------------------------------
conduit_index_t
conduit_node_number_of_children(conduit_node *cnode)
{
    return cpp_node(cnode)->number_of_children();
}

//-----------------------------------------------------------------------------
conduit_index_t
conduit_node_number_of_elements(conduit_node *cnode)
{
    return cpp_node(cnode)->dtype().number_of_elements();
}

//-----------------------------------------------------------------------------
int 
conduit_node_has_child(const conduit_node *cnode, 
                       const char *name)
{
    return (int)cpp_node(cnode)->has_child(std::string(name));
}

//-----------------------------------------------------------------------------
int
conduit_node_has_path(const conduit_node *cnode, 
                      const char *path)
{
    return (int)cpp_node(cnode)->has_path(std::string(path));
}

//-----------------------------------------------------------------------------
int
conduit_node_is_root(conduit_node *cnode)
{
    return (int)cpp_node(cnode)->is_root();
}

//-----------------------------------------------------------------------------
int
conduit_node_is_data_external(const conduit_node *cnode)
{
    return (int)cpp_node(cnode)->is_data_external();
}


//-----------------------------------------------------------------------------
conduit_node * 
conduit_node_parent(conduit_node *cnode)
{
    return c_node(cpp_node(cnode)->parent());
}

//-----------------------------------------------------------------------------
conduit_index_t
conduit_node_total_strided_bytes(const conduit_node *cnode)
{
    return cpp_node(cnode)->total_strided_bytes();
}

//-----------------------------------------------------------------------------
conduit_index_t
conduit_node_total_bytes_compact(const conduit_node *cnode)
{
    return cpp_node(cnode)->total_bytes_compact();
}

//-----------------------------------------------------------------------------
int
conduit_node_is_compact(const conduit_node *cnode)
{
    return (int)cpp_node(cnode)->is_compact();
}

//-----------------------------------------------------------------------------
int 
conduit_node_is_contiguous(const conduit_node *cnode)
{
    return (int)cpp_node(cnode)->is_contiguous();
}

//-----------------------------------------------------------------------------
int 
conduit_node_contiguous_with_node(const conduit_node *cnode,
                                  const conduit_node *cother)
{
    return (int)cpp_node(cnode)->contiguous_with(cpp_node_ref(cother));
}

//-----------------------------------------------------------------------------
int 
conduit_node_contiguous_with_address(const conduit_node *cnode,
                                     void *address)
{
    return (int)cpp_node(cnode)->contiguous_with(address);
}


//-----------------------------------------------------------------------------
int
conduit_node_diff(const conduit_node *cnode,
                  const conduit_node *cother,
                  conduit_node *cinfo,
                  conduit_float64 epsilon)
{
    return (int) cpp_node(cnode)->diff(cpp_node_ref(cother),
                                       cpp_node_ref(cinfo),
                                       epsilon);
}


//-----------------------------------------------------------------------------
int
conduit_node_diff_compatible(const conduit_node *cnode,
                             const conduit_node *cother,
                             conduit_node *cinfo,
                             conduit_float64 epsilon)
{
    return (int) cpp_node(cnode)->diff_compatible(cpp_node_ref(cother),
                                                  cpp_node_ref(cinfo),
                                                  epsilon);
}

//-----------------------------------------------------------------------------
void
conduit_node_info(const conduit_node *cnode,
                  const conduit_node *cnres)
{
    cpp_node(cnode)->compatible(cpp_node_ref(cnres));
}

//-----------------------------------------------------------------------------
void 
conduit_node_print(conduit_node *cnode)
{
    cpp_node(cnode)->print();
}

//-----------------------------------------------------------------------------
void 
conduit_node_print_detailed(conduit_node *cnode)
{
    cpp_node(cnode)->print_detailed();
}





//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set variants -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for generic types --
//-----------------------------------------------------------------------------
void
conduit_set_node(conduit_node *cnode,
                 conduit_node *data)
{
    cpp_node(cnode)->set_node(*cpp_node(data));
}


//-----------------------------------------------------------------------------
// -- set path for generic types --
//-----------------------------------------------------------------------------
void
conduit_set_path_node(conduit_node *cnode,
                      const char* path,
                      conduit_node *data)
{
    cpp_node(cnode)->set_path_node(path,*cpp_node(data));
}

//-----------------------------------------------------------------------------
// -- set external for generic types --
//-----------------------------------------------------------------------------
void
conduit_set_external_node(conduit_node *cnode,
                          conduit_node *data)
{
    cpp_node(cnode)->set_external_node(*cpp_node(data));
}


//-----------------------------------------------------------------------------
// -- set path external for generic types --
//-----------------------------------------------------------------------------
void
conduit_set_path_external_node(conduit_node *cnode,
                               const char* path,
                               conduit_node *data)
{
    cpp_node(cnode)->set_path_external_node(path,*cpp_node(data));
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// string cases
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// string set 
//-----------------------------------------------------------------------------
void
conduit_node_set_char8_str(conduit_node *cnode, 
                           const char *value)
{
    cpp_node(cnode)->set_char8_str(value);
}

//-----------------------------------------------------------------------------
// string set_path
//-----------------------------------------------------------------------------
void
conduit_node_set_path_char8_str(conduit_node *cnode,
                                const char *path,
                                const char *value)
{
    cpp_node(cnode)->set_path_char8_str(path,value);
}


//-----------------------------------------------------------------------------
// string set_external
//-----------------------------------------------------------------------------
void
conduit_node_set_external_char8_str(conduit_node *cnode, 
                                    char *value)
{
    cpp_node(cnode)->set_external_char8_str(value);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set for scalar bitwidth style types ---
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// set bitwidth signed integer scalar types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
conduit_node_set_int8(conduit_node *cnode,
                      conduit_int8 value)
{
    cpp_node(cnode)->set_int8(value);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_int16(conduit_node *cnode,
                       conduit_int16 value)
{
    cpp_node(cnode)->set_int16(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int32(conduit_node *cnode,
                       conduit_int32 value)
{
    cpp_node(cnode)->set_int32(value);    
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int64(conduit_node *cnode,
                       conduit_int64 value)
{
    cpp_node(cnode)->set_int64(value);
}


//-----------------------------------------------------------------------------
// set bitwidth unsigned integer scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_uint8(conduit_node *cnode,
                       conduit_uint8 value)
{
    cpp_node(cnode)->set_uint8(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_uint16(conduit_node *cnode,
                        conduit_uint16 value)
{
    cpp_node(cnode)->set_uint16(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_uint32(conduit_node *cnode,
                        conduit_uint32 value)
{
    cpp_node(cnode)->set_uint32(value);
}

void
conduit_node_set_uint64(conduit_node *cnode,
                        conduit_uint64 value)
{
    cpp_node(cnode)->set_uint64(value);
}


//-----------------------------------------------------------------------------
// set bitwidth floating point scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_float32(conduit_node *cnode,
                         conduit_float32 value)
{
    cpp_node(cnode)->set_float32(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_float64(conduit_node *cnode,
                         conduit_float64 value)
{
    cpp_node(cnode)->set_float64(value);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set using bitwidth style pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// set bitwidth signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_int8_ptr(conduit_node *cnode,
                          conduit_int8 *data,
                          conduit_index_t num_elements)
{
    cpp_node(cnode)->set_int8_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int8_ptr_detailed(conduit_node *cnode,
                                   conduit_int8 *data,
                                   conduit_index_t num_elements,
                                   conduit_index_t offset,
                                   conduit_index_t stride,
                                   conduit_index_t element_bytes,
                                   conduit_index_t endianness)
{
    cpp_node(cnode)->set_int8_ptr(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int16_ptr(conduit_node *cnode,
                           conduit_int16 *data,
                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set_int16_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int16_ptr_detailed(conduit_node *cnode,
                                    conduit_int16 *data,
                                    conduit_index_t num_elements,
                                    conduit_index_t offset,
                                    conduit_index_t stride,
                                    conduit_index_t element_bytes,
                                    conduit_index_t endianness)
{
    cpp_node(cnode)->set_int16_ptr(data,
                                   num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int32_ptr(conduit_node *cnode,
                           conduit_int32 *data,
                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set_int32_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int32_ptr_detailed(conduit_node *cnode,
                                    conduit_int32 *data,
                                    conduit_index_t num_elements,
                                    conduit_index_t offset,
                                    conduit_index_t stride,
                                    conduit_index_t element_bytes,
                                    conduit_index_t endianness)
{
    cpp_node(cnode)->set_int32_ptr(data,
                                   num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int64_ptr(conduit_node *cnode,
                           conduit_int64 *data,
                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set_int64_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int64_ptr_detailed(conduit_node *cnode,
                                    conduit_int64 *data,
                                    conduit_index_t num_elements,
                                    conduit_index_t offset,
                                    conduit_index_t stride,
                                    conduit_index_t element_bytes,
                                    conduit_index_t endianness)
{
    cpp_node(cnode)->set_int64_ptr(data,
                                   num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness);
}


//-----------------------------------------------------------------------------
// set bitwidth unsigned signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_uint8_ptr(conduit_node *cnode,
                           conduit_uint8 *data,
                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set_uint8_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_uint8_ptr_detailed(conduit_node *cnode,
                                         conduit_uint8 *data,
                                         conduit_index_t num_elements,
                                         conduit_index_t offset,
                                         conduit_index_t stride,
                                         conduit_index_t element_bytes,
                                         conduit_index_t endianness)
{
    cpp_node(cnode)->set_uint8_ptr(data,
                                   num_elements,
                                   offset,
                                   stride,
                                   element_bytes,
                                   endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_uint16_ptr(conduit_node *cnode,
                            conduit_uint16 *data,
                            conduit_index_t num_elements)
{
    cpp_node(cnode)->set_uint16_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_uint16_ptr_detailed(conduit_node *cnode,
                                     conduit_uint16 *data,
                                     conduit_index_t num_elements,
                                     conduit_index_t offset,
                                     conduit_index_t stride,
                                     conduit_index_t element_bytes,
                                     conduit_index_t endianness)
{
    cpp_node(cnode)->set_uint16_ptr(data,
                                    num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_uint32_ptr(conduit_node *cnode,
                            conduit_uint32 *data,
                            conduit_index_t num_elements)
{
    cpp_node(cnode)->set_uint32_ptr(data,num_elements);
}
    
//-----------------------------------------------------------------------------
void
conduit_node_set_uint32_ptr_detailed(conduit_node *cnode,
                                     conduit_uint32 *data,
                                     conduit_index_t num_elements,
                                     conduit_index_t offset,
                                     conduit_index_t stride,
                                     conduit_index_t element_bytes,
                                     conduit_index_t endianness)
{
    cpp_node(cnode)->set_uint32_ptr(data,
                                    num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_uint64_ptr(conduit_node *cnode,
                            conduit_uint64 *data,
                            conduit_index_t num_elements)
{
    cpp_node(cnode)->set_uint64_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_uint64_ptr_detailed(conduit_node *cnode,
                                     conduit_uint64 *data,
                                     conduit_index_t num_elements,
                                     conduit_index_t offset,
                                     conduit_index_t stride,
                                     conduit_index_t element_bytes,
                                     conduit_index_t endianness)
{
    cpp_node(cnode)->set_uint64_ptr(data,
                                    num_elements,
                                    offset,
                                    stride,
                                    element_bytes,
                                    endianness);
}

//-----------------------------------------------------------------------------
// set bitwidth floating point pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_float32_ptr(conduit_node *cnode,
                             conduit_float32 *data,
                             conduit_index_t num_elements)
{
    cpp_node(cnode)->set_float32_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_float32_ptr_detailed(conduit_node *cnode,
                                      conduit_float32 *data,
                                      conduit_index_t num_elements,
                                      conduit_index_t offset,
                                      conduit_index_t stride,
                                      conduit_index_t element_bytes,
                                      conduit_index_t endianness)
{
    cpp_node(cnode)->set_float32_ptr(data,
                                     num_elements,
                                     offset,
                                     stride,
                                     element_bytes,
                                     endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_float64_ptr(conduit_node *cnode,
                             conduit_float64 *data,
                             conduit_index_t  num_elements)
{
    cpp_node(cnode)->set_float64_ptr(data,num_elements);
}


//-------------------------------------------------------------------------
void
conduit_node_set_float64_ptr_detailed(conduit_node *cnode,
                                      conduit_float64 *data,
                                      conduit_index_t num_elements,
                                      conduit_index_t offset,
                                      conduit_index_t stride,
                                      conduit_index_t element_bytes,
                                      conduit_index_t endianness)
{
    cpp_node(cnode)->set_float64_ptr(data,
                                     num_elements,
                                     offset,
                                     stride,
                                     element_bytes,
                                     endianness);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set_path variants -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set path for scalar types ---
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// set_path bitwidth signed integer scalar types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------s
void
conduit_node_set_path_int8(conduit_node *cnode,
                           const char *path,
                           conduit_int8 value)
{
    cpp_node(cnode)->set_path_int8(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int16(conduit_node *cnode,
                            const char *path,
                            conduit_int16 value)
{
    cpp_node(cnode)->set_path_int16(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int32(conduit_node *cnode,
                            const char *path,
                            conduit_int32 value)
{
    cpp_node(cnode)->set_path_int32(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int64(conduit_node *cnode,
                            const char *path,
                            conduit_int64 value)
{
    cpp_node(cnode)->set_path_int64(path,value);
}

//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------
// set_path bitwidth unsigned integer scalar types
//-------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint8(conduit_node *cnode,
                            const char *path,
                            conduit_uint8 value)
{
    cpp_node(cnode)->set_path_uint8(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint16(conduit_node *cnode,
                             const char *path,
                             conduit_uint16 value)
{
    cpp_node(cnode)->set_path_uint16(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint32(conduit_node *cnode,
                             const char *path,
                             conduit_uint32 value)
{
    cpp_node(cnode)->set_path_uint32(path,value);
}
    
//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint64(conduit_node *cnode,
                             const char *path,
                             conduit_uint64 value)
{    
    cpp_node(cnode)->set_path_uint64(path,value);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// set_path bitwidth floating point scalar types
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
conduit_node_set_path_float32(conduit_node *cnode,
                              const char *path,
                              conduit_float32 value)
{
    cpp_node(cnode)->set_path_float32(path,value);
}
    
//-----------------------------------------------------------------------------
void
conduit_node_set_path_float64(conduit_node *cnode,
                              const char *path,
                              conduit_float64 value)
{
    cpp_node(cnode)->set_path_float64(path,value);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set path using bitwidth style pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// set_path bitwidth signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_int8_ptr(conduit_node *cnode,
                               const char *path,
                               conduit_int8 *data,
                               conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_int8_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int8_ptr_detailed(conduit_node *cnode,
                                        const char *path,
                                        conduit_int8 *data,
                                        conduit_index_t num_elements,
                                        conduit_index_t offset,
                                        conduit_index_t stride,
                                        conduit_index_t element_bytes,
                                        conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_int8_ptr(path,
                                       data,
                                       num_elements,
                                       offset,
                                       stride,
                                       element_bytes,
                                       endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int16_ptr(conduit_node *cnode,
                                const char *path,
                                conduit_int16 *data,
                                conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_int16_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int16_ptr_detailed(conduit_node *cnode,
                                         const char *path,
                                         conduit_int16 *data,
                                         conduit_index_t num_elements,
                                         conduit_index_t offset,
                                         conduit_index_t stride,
                                         conduit_index_t element_bytes,
                                         conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_int16_ptr(path,
                                        data,
                                        num_elements,
                                        offset,
                                        stride,
                                        element_bytes,
                                        endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int32_ptr(conduit_node *cnode,
                                const char *path,
                                conduit_int32 *data,
                                conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_int32_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int32_ptr_detailed(conduit_node *cnode,
                                         const char *path,
                                         conduit_int32 *data,
                                         conduit_index_t num_elements,
                                         conduit_index_t offset,
                                         conduit_index_t stride,
                                         conduit_index_t element_bytes,
                                         conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_int32_ptr(path,
                                        data,
                                        num_elements,
                                        offset,
                                        stride,
                                        element_bytes,
                                        endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int64_ptr(conduit_node *cnode,
                                const char *path,
                                conduit_int64 *data,
                                conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_int64_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int64_ptr_detailed(conduit_node *cnode,
                                         const char *path,
                                         conduit_int64 *data,
                                         conduit_index_t num_elements,
                                         conduit_index_t offset,
                                         conduit_index_t stride,
                                         conduit_index_t element_bytes,
                                         conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_int64_ptr(path,
                                        data,
                                        num_elements,
                                        offset,
                                        stride,
                                        element_bytes,
                                        endianness);
}


//-----------------------------------------------------------------------------
// set_path bitwidth unsigned signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint8_ptr(conduit_node *cnode,
                                const char *path,
                                conduit_uint8 *data,
                                conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_uint8_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_path_uint8_ptr_detailed(conduit_node *cnode,
                                              const char *path,
                                              conduit_uint8 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_uint8_ptr(path,
                                        data,
                                        num_elements,
                                        offset,
                                        stride,
                                        element_bytes,
                                        endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint16_ptr(conduit_node *cnode,
                                 const char *path,
                                 conduit_uint16 *data,
                                 conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_uint16_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint16_ptr_detailed(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint16 *data,
                                          conduit_index_t num_elements,
                                          conduit_index_t offset,
                                          conduit_index_t stride,
                                          conduit_index_t element_bytes,
                                          conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_uint16_ptr(path,
                                         data,
                                         num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint32_ptr(conduit_node *cnode,
                                 const char *path,
                                 conduit_uint32 *data,
                                 conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_uint32_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint32_ptr_detailed(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint32 *data,
                                          conduit_index_t num_elements,
                                          conduit_index_t offset,
                                          conduit_index_t stride,
                                          conduit_index_t element_bytes,
                                          conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_uint32_ptr(path,
                                         data,
                                         num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint64_ptr(conduit_node *cnode,
                                 const char *path,
                                 conduit_uint64 *data,
                                 conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_uint64_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_uint64_ptr_detailed(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint64 *data,
                                          conduit_index_t num_elements,
                                          conduit_index_t offset,
                                          conduit_index_t stride,
                                          conduit_index_t element_bytes,
                                          conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_uint64_ptr(path,
                                         data,
                                         num_elements,
                                         offset,
                                         stride,
                                         element_bytes,
                                         endianness);
}

//-----------------------------------------------------------------------------
// set_path bitwidth floating point pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_float32_ptr(conduit_node *cnode,
                                  const char *path,
                                  conduit_float32 *data,
                                  conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_float32_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_float32_ptr_detailed(conduit_node *cnode,
                                           const char *path,
                                           conduit_float32 *data,
                                           conduit_index_t num_elements,
                                           conduit_index_t offset,
                                           conduit_index_t stride,
                                           conduit_index_t element_bytes,
                                           conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_float32_ptr(path,
                                          data,
                                          num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_float64_ptr(conduit_node *cnode,
                                  const char *path,
                                  conduit_float64 *data,
                                  conduit_index_t  num_elements)
{
    cpp_node(cnode)->set_path_float64_ptr(path,data,num_elements);
}


//-------------------------------------------------------------------------
void
conduit_node_set_path_float64_ptr_detailed(conduit_node *cnode,
                                           const char *path,
                                           conduit_float64 *data,
                                           conduit_index_t num_elements,
                                           conduit_index_t offset,
                                           conduit_index_t stride,
                                           conduit_index_t element_bytes,
                                           conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_float64_ptr(path,
                                          data,
                                          num_elements,
                                          offset,
                                          stride,
                                          element_bytes,
                                          endianness);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set external using bitwidth style pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// set_external bitwdith signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_external_int8_ptr(conduit_node *cnode,
                                   conduit_int8 *data,
                                   conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_int8_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_int8_ptr_detailed(conduit_node *cnode,
                                            conduit_int8 *data,
                                            conduit_index_t num_elements,
                                            conduit_index_t offset,
                                            conduit_index_t stride,
                                            conduit_index_t element_bytes,
                                            conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_int8_ptr(data,
                                           num_elements,
                                           offset,
                                           stride,
                                           element_bytes,
                                           endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_extenal_int16_ptr(conduit_node *cnode,
                                   conduit_int16 *data,
                                   conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_int16_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_int16_ptr_detailed(conduit_node *cnode,
                                             conduit_int16 *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_int16_ptr(data,
                                            num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_int32_ptr(conduit_node *cnode,
                                    conduit_int32 *data,
                                    conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_int32_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_int32_ptr_detailed(conduit_node *cnode,
                                             conduit_int32 *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_int32_ptr(data,
                                            num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_int64_ptr(conduit_node *cnode,
                           conduit_int64 *data,
                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_int64_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_int64_ptr_detailed(conduit_node *cnode,
                                             conduit_int64 *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_int64_ptr(data,
                                            num_elements,
                                            offset, 
                                            stride,
                                            element_bytes,
                                            endianness);
}


//-----------------------------------------------------------------------------
// set_external bitwidth unsigned signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_external_uint8_ptr(conduit_node *cnode,
                                    conduit_uint8 *data,
                                    conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_uint8_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_uint8_ptr_detailed(conduit_node *cnode,
                                                  conduit_uint8 *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_uint8_ptr(data,
                                            num_elements,
                                            offset,
                                            stride,
                                            element_bytes,
                                            endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_uint16_ptr(conduit_node *cnode,
                                     conduit_uint16 *data,
                                     conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_uint16_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_uint16_ptr_detailed(conduit_node *cnode,
                                              conduit_uint16 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_uint16_ptr(data,
                                             num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_uint32_ptr(conduit_node *cnode,
                                     conduit_uint32 *data,
                                     conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_uint32_ptr(data,num_elements);
}
    
//-----------------------------------------------------------------------------
void
conduit_node_set_external_uint32_ptr_detailed(conduit_node *cnode,
                                              conduit_uint32 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_uint32_ptr(data,
                                             num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_uint64_ptr(conduit_node *cnode,
                                     conduit_uint64 *data,
                                     conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_uint64_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_uint64_ptr_detailed(conduit_node *cnode,
                                              conduit_uint64 *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_uint64_ptr(data,
                                             num_elements,
                                             offset,
                                             stride,
                                             element_bytes,
                                             endianness);
}

//-----------------------------------------------------------------------------
// set_external bitwdith floating point pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_external_float32_ptr(conduit_node *cnode,
                                      conduit_float32 *data,
                                      conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external_float32_ptr(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_float32_ptr_detailed(conduit_node *cnode,
                                               conduit_float32 *data,
                                               conduit_index_t num_elements,
                                               conduit_index_t offset,
                                               conduit_index_t stride,
                                               conduit_index_t element_bytes,
                                               conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_float32_ptr(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_external_float64_ptr(conduit_node *cnode,
                                      conduit_float64 *data,
                                      conduit_index_t  num_elements)
{
    cpp_node(cnode)->set_external_float64_ptr(data,num_elements);
}


//-------------------------------------------------------------------------
void
conduit_node_set_external_float64_ptr_detailed(conduit_node *cnode,
                                               conduit_float64 *data,
                                               conduit_index_t num_elements,
                                               conduit_index_t offset,
                                               conduit_index_t stride,
                                               conduit_index_t element_bytes,
                                               conduit_index_t endianness)
{
    cpp_node(cnode)->set_external_float64_ptr(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set_path external variants -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// set_external bitwdith signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int8_ptr(conduit_node *cnode,
                                        const char *path,
                                        conduit_int8 *data,
                                        conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_int8_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int8_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 conduit_int8 *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_int8_ptr(path,
                                                data,
                                                num_elements,
                                                offset,
                                                stride,
                                                element_bytes,
                                                endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int16_ptr(conduit_node *cnode,
                                         const char *path,
                                         conduit_int16 *data,
                                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_int16_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int16_ptr_detailed(conduit_node *cnode,
                                                  const char *path,
                                                  conduit_int16 *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_int16_ptr(path,
                                                 data,
                                                 num_elements,
                                                 offset,
                                                 stride,
                                                 element_bytes,
                                                 endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int32_ptr(conduit_node *cnode,
                                         const char *path,
                                         conduit_int32 *data,
                                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_int32_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int32_ptr_detailed(conduit_node *cnode,
                                                  const char *path,
                                                  conduit_int32 *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_int32_ptr(path,
                                                 data,
                                                 num_elements,
                                                 offset,
                                                 stride,
                                                 element_bytes,
                                                 endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int64_ptr(conduit_node *cnode,
                                         const char *path,
                                         conduit_int64 *data,
                                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_int64_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int64_ptr_detailed(conduit_node *cnode,
                                                  const char *path,
                                                  conduit_int64 *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_int64_ptr(path,
                                                 data,
                                                 num_elements,
                                                 offset, 
                                                 stride,
                                                 element_bytes,
                                                 endianness);
}


//-----------------------------------------------------------------------------
// set_path_external bitwdith unsigned signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_uint8_ptr(conduit_node *cnode,
                                         const char *path,
                                         conduit_uint8 *data,
                                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_uint8_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_path_external_uint8_ptr_detailed(conduit_node *cnode,
                                                       const char *path,
                                                       conduit_uint8 *data,
                                                       conduit_index_t num_elements,
                                                       conduit_index_t offset,
                                                       conduit_index_t stride,
                                                       conduit_index_t element_bytes,
                                                       conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_uint8_ptr(path,
                                                 data,
                                                 num_elements,
                                                 offset,
                                                 stride,
                                                 element_bytes,
                                                 endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_uint16_ptr(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint16 *data,
                                          conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_uint16_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_uint16_ptr_detailed(conduit_node *cnode,
                                                   const char *path,
                                                   conduit_uint16 *data,
                                                   conduit_index_t num_elements,
                                                   conduit_index_t offset,
                                                   conduit_index_t stride,
                                                   conduit_index_t element_bytes,
                                                   conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_uint16_ptr(path,
                                                  data,
                                                  num_elements,
                                                  offset,
                                                  stride,
                                                  element_bytes,
                                                  endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_uint32_ptr(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint32 *data,
                                          conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_uint32_ptr(path,data,num_elements);
}
    
//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_uint32_ptr_detailed(conduit_node *cnode,
                                                   const char *path,
                                                   conduit_uint32 *data,
                                                   conduit_index_t num_elements,
                                                   conduit_index_t offset,
                                                   conduit_index_t stride,
                                                   conduit_index_t element_bytes,
                                                   conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_uint32_ptr(path,
                                                  data,
                                                  num_elements,
                                                  offset,
                                                  stride,
                                                  element_bytes,
                                                  endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_uint64_ptr(conduit_node *cnode,
                                          const char *path,
                                          conduit_uint64 *data,
                                          conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_uint64_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_uint64_ptr_detailed(conduit_node *cnode,
                                                   const char *path,
                                                   conduit_uint64 *data,
                                                   conduit_index_t num_elements,
                                                   conduit_index_t offset,
                                                   conduit_index_t stride,
                                                   conduit_index_t element_bytes,
                                                   conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_uint64_ptr(path,
                                                  data,
                                                  num_elements,
                                                  offset,
                                                  stride,
                                                  element_bytes,
                                                  endianness);
}

//-----------------------------------------------------------------------------
// set_path_external bitwdith floating point pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_float32_ptr(conduit_node *cnode,
                                           const char *path,
                                           conduit_float32 *data,
                                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path_external_float32_ptr(path,data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_float32_ptr_detailed(conduit_node *cnode,
                                                    const char *path,
                                                    conduit_float32 *data,
                                                    conduit_index_t num_elements,
                                                    conduit_index_t offset,
                                                    conduit_index_t stride,
                                                    conduit_index_t element_bytes,
                                                    conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_float32_ptr(path,
                                                   data,
                                                   num_elements,
                                                   offset,
                                                   stride,
                                                   element_bytes,
                                                   endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_float64_ptr(conduit_node *cnode,
                                           const char *path,
                                           conduit_float64 *data,
                                           conduit_index_t  num_elements)
{
    cpp_node(cnode)->set_path_external_float64_ptr(path,data,num_elements);
}


//-------------------------------------------------------------------------
void
conduit_node_set_path_external_float64_ptr_detailed(conduit_node *cnode,
                                                    const char *path,
                                                    conduit_float64 *data,
                                                    conduit_index_t num_elements,
                                                    conduit_index_t offset,
                                                    conduit_index_t stride,
                                                    conduit_index_t element_bytes,
                                                    conduit_index_t endianness)
{
    cpp_node(cnode)->set_path_external_float64_ptr(path,
                                                   data,
                                                   num_elements,
                                                   offset,
                                                   stride,
                                                   element_bytes,
                                                   endianness);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// -- set for cstyle style types -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- set for scalar cstyle style types ---
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// set cstyle signed integer scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_char(conduit_node *cnode,
                      char value)
{
    cpp_node(cnode)->set((conduit_byte)value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_short(conduit_node *cnode,
                       short value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_int(conduit_node *cnode,
                     int value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_long(conduit_node *cnode,
                      long value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
// set cstyle style unsigned integer scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_unsigned_char(conduit_node *cnode,
                               unsigned char value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_unsigned_short(conduit_node *cnode,
                                unsigned short value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_unsigned_int(conduit_node *cnode,
                                            unsigned int value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_unsigned_long(conduit_node *cnode,
                               unsigned long value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
// set cstyle style floating point scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_float(conduit_node *cnode,
                       float value)
{
    cpp_node(cnode)->set(value);
}

void
conduit_node_set_double(conduit_node *cnode,
                        double value)
{
    cpp_node(cnode)->set(value);
}

//-----------------------------------------------------------------------------
// -- set via cstyle style pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
//-------------------------------------------------------------------------
// set cstyle signed integer pointer cases
//-------------------------------------------------------------------------
void
conduit_node_set_char_ptr(conduit_node *cnode,
                          char *data,
                          conduit_index_t num_elements)
{
    cpp_node(cnode)->set((conduit_byte*)data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_char_ptr_detailed(conduit_node *cnode,
                                   char *data,
                                   conduit_index_t num_elements,
                                   conduit_index_t offset,
                                   conduit_index_t stride,
                                   conduit_index_t element_bytes,
                                   conduit_index_t endianness)
{
    cpp_node(cnode)->set((conduit_byte*)data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_short_ptr(conduit_node *cnode,
                           short *data,
                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_short_ptr_detailed(conduit_node *cnode,
                                    short *data,
                                    conduit_index_t num_elements,
                                    conduit_index_t offset,
                                    conduit_index_t stride,
                                    conduit_index_t element_bytes,
                                    conduit_index_t endianness)
{
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_int_ptr(conduit_node *cnode,
                         int *data,
                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_int_ptr_detailed(conduit_node *cnode,
                                  int *data,
                                  conduit_index_t num_elements,
                                  conduit_index_t offset,
                                  conduit_index_t stride,
                                  conduit_index_t element_bytes,
                                  conduit_index_t endianness)
{    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_long_ptr(conduit_node *cnode,
                          long *data,
                          conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}


//-------------------------------------------------------------------------
void
conduit_node_set_long_ptr_detailed(conduit_node *cnode,
                                   long *data,
                                   conduit_index_t num_elements,
                                   conduit_index_t offset,
                                   conduit_index_t stride,
                                   conduit_index_t element_bytes,
                                   conduit_index_t endianness)
{    
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}


//-------------------------------------------------------------------------
// set cstyle unsigned signed integer pointer cases
//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_char_ptr(conduit_node *cnode,
                                   unsigned char *data,
                                   conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_char_ptr_detailed(conduit_node *cnode,
                                            unsigned char *data,
                                            conduit_index_t num_elements,
                                            conduit_index_t offset,
                                            conduit_index_t stride,
                                            conduit_index_t element_bytes,
                                            conduit_index_t endianness)
{    
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_short_ptr(conduit_node *cnode,
                                    unsigned short *data,
                                    conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_short_ptr_detailed(conduit_node *cnode,
                                             unsigned short *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness)
{    
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_int_ptr(conduit_node *cnode,
                                  unsigned int *data,
                                  conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_int_ptr_detailed(conduit_node *cnode,
                                           unsigned int *data,
                                           conduit_index_t num_elements,
                                           conduit_index_t offset,
                                           conduit_index_t stride,
                                           conduit_index_t element_bytes,
                                           conduit_index_t endianness)
{    
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_long_ptr(conduit_node *cnode,
                                   unsigned long *data,
                                   conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_unsigned_long_ptr_detailed(conduit_node *cnode,
                                            unsigned long *data,
                                            conduit_index_t num_elements,
                                            conduit_index_t offset,
                                            conduit_index_t stride,
                                            conduit_index_t element_bytes,
                                            conduit_index_t endianness)
{    
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}

//-------------------------------------------------------------------------
// set cstyle floating point pointer cases
//-------------------------------------------------------------------------
void
conduit_node_set_float_ptr(conduit_node *cnode,
                           float *data,
                           conduit_index_t num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_float_ptr_detailed(conduit_node *cnode,
                                    float *data,
                                    conduit_index_t num_elements,
                                    conduit_index_t offset,
                                    conduit_index_t stride,
                                    conduit_index_t element_bytes,
                                    conduit_index_t endianness)
{    
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}


//-------------------------------------------------------------------------
void
conduit_node_set_double_ptr(conduit_node *cnode,
                            double *data,
                            conduit_index_t  num_elements)
{
    cpp_node(cnode)->set(data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_double_ptr_detailed(conduit_node *cnode,
                                     double *data,
                                     conduit_index_t num_elements,
                                     conduit_index_t offset,
                                     conduit_index_t stride,
                                     conduit_index_t element_bytes,
                                     conduit_index_t endianness)
{    
    cpp_node(cnode)->set(data,
                         num_elements,
                         offset,
                         stride,
                         element_bytes,
                         endianness);
}


//-----------------------------------------------------------------------------
// -- set_path for scalar cstyle types ---
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// set_path cstyle signed integer scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_path_char(conduit_node *cnode,
                           const char *path,
                           char value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_short(conduit_node *cnode,
                            const char *path,
                            short value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_int(conduit_node *cnode,
                          const char *path,
                          int value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_long(conduit_node *cnode,
                           const char *path,
                           long value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
// set_path cstyle unsigned integer scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_char(conduit_node *cnode,
                                    const char *path,
                                    unsigned char value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_short(conduit_node *cnode,
                                     const char *path,
                                     unsigned short value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_int(conduit_node *cnode,
                                   const char *path,
                                   unsigned int value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_long(conduit_node *cnode,
                                    const char *path,
                                    unsigned long value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
// set_path cstyle floating point scalar types
//-----------------------------------------------------------------------------
void
conduit_node_set_path_float(conduit_node *cnode,
                            const char *path,
                            float value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_double(conduit_node *cnode,
                             const char *path,
                             double value)
{
    cpp_node(cnode)->set_path(path,value);
}

//-----------------------------------------------------------------------------
// -- set_path via cstyle pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// set_path cstyle signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_char_ptr(conduit_node *cnode,
                               const char *path,
                               char *data,
                               conduit_index_t num_elements)

{
    cpp_node(cnode)->set_path(path,(conduit_byte*)data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_char_ptr_detailed(conduit_node *cnode,
                                        const char *path,
                                        char *data,
                                        conduit_index_t num_elements,
                                        conduit_index_t offset,
                                        conduit_index_t stride,
                                        conduit_index_t element_bytes,
                                        conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              (conduit_byte*)data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_short_ptr(conduit_node *cnode,
                                const char *path,
                                short *data,
                                conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_short_ptr_detailed(conduit_node *cnode,
                                         const char *path,
                                         short *data,
                                         conduit_index_t num_elements,
                                         conduit_index_t offset,
                                         conduit_index_t stride,
                                         conduit_index_t element_bytes,
                                         conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_int_ptr(conduit_node *cnode,
                              const char *path,
                              int *data,
                              conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_int_ptr_detailed(conduit_node *cnode,
                                       const char *path,
                                       int *data,
                                       conduit_index_t num_elements,
                                       conduit_index_t offset,
                                       conduit_index_t stride,
                                       conduit_index_t element_bytes,
                                       conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_long_ptr(conduit_node *cnode,
                               const char *path,
                               long *data,
                               conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_long_ptr_detailed(conduit_node *cnode,
                                        const char *path,
                                        long *data,
                                        conduit_index_t num_elements,
                                        conduit_index_t offset,
                                        conduit_index_t stride,
                                        conduit_index_t element_bytes,
                                        conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
// set_path cstyle unsigned integer pointer cases
//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_char_ptr(conduit_node *cnode,
                                        const char *path,
                                        unsigned char *data,
                                        conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_char_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 unsigned char *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_short_ptr(conduit_node *cnode,
                                         const char *path,
                                         unsigned short *data,
                                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_short_ptr_detailed(conduit_node *cnode,
                                                  const char *path,
                                                  unsigned short *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_int_ptr(conduit_node *cnode,
                                       const char *path,
                                       unsigned int *data,
                                       conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_int_ptr_detailed(conduit_node *cnode,
                                                const char *path,
                                                unsigned int *data,
                                                conduit_index_t num_elements,
                                                conduit_index_t offset,
                                                conduit_index_t stride,
                                                conduit_index_t element_bytes,
                                                conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_long_ptr(conduit_node *cnode,
                                        const char *path,
                                        unsigned long *data,
                                        conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_unsigned_long_ptr_detailed(conduit_node *cnode,
                                                 const char *path,
                                                 unsigned long *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
// set_path cstyle floating point pointer cases
//-------------------------------------------------------------------------
void
conduit_node_set_path_float_ptr(conduit_node *cnode,
                                const char *path,
                                float *data,
                                conduit_index_t num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_float_ptr_detailed(conduit_node *cnode,
                                         const char *path,
                                         float *data,
                                         conduit_index_t num_elements,
                                         conduit_index_t offset,
                                         conduit_index_t stride,
                                         conduit_index_t element_bytes,
                                         conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_double_ptr(conduit_node *cnode,
                                 const char *path,
                                 double *data,
                                 conduit_index_t  num_elements)
{
    cpp_node(cnode)->set_path(path,data,num_elements);
}

//-------------------------------------------------------------------------
void
conduit_node_set_path_double_ptr_detailed(conduit_node *cnode,
                                          const char *path,
                                          double *data,
                                          conduit_index_t num_elements,
                                          conduit_index_t offset,
                                          conduit_index_t stride,
                                          conduit_index_t element_bytes,
                                          conduit_index_t endianness)
{    
    cpp_node(cnode)->set_path(path,
                              data,
                              num_elements,
                              offset,
                              stride,
                              element_bytes,
                              endianness);
}


//-----------------------------------------------------------------------------
// -- set_external via cstyle pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// set_external cstyle signed integer pointer cases
//-----------------------------------------------------------------------------
void conduit_node_set_external_char_ptr(conduit_node *cnode,
                                        char *data,
                                        conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external((conduit_byte*)data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_char_ptr_detailed(conduit_node *cnode,
                                                 char *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external((conduit_byte*)data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_short_ptr(conduit_node *cnode,
                                         short *data,
                                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_short_ptr_detailed(conduit_node *cnode,
                                                  short *data,
                                                  conduit_index_t num_elements,
                                                  conduit_index_t offset,
                                                  conduit_index_t stride,
                                                  conduit_index_t element_bytes,
                                                  conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_int_ptr(conduit_node *cnode,
                                       int *data,
                                       conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_int_ptr_detailed(conduit_node *cnode,
                                                int *data,
                                                conduit_index_t num_elements,
                                                conduit_index_t offset,
                                                conduit_index_t stride,
                                                conduit_index_t element_bytes,
                                                conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}


//-----------------------------------------------------------------------------
void conduit_node_set_external_long_ptr(conduit_node *cnode,
                                        long *data,
                                        conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_long_ptr_detailed(conduit_node *cnode,
                                                 long *data,
                                                 conduit_index_t num_elements,
                                                 conduit_index_t offset,
                                                 conduit_index_t stride,
                                                 conduit_index_t element_bytes,
                                                 conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}


//-----------------------------------------------------------------------------
// set_external cstyle unsigned integer pointer cases
//-----------------------------------------------------------------------------
void conduit_node_set_external_unsigned_char_ptr(conduit_node *cnode,
                                                 unsigned char *data,
                                                 conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void conduit_node_set_external_unsigned_char_ptr_detailed(conduit_node *cnode,
                                                          unsigned char *data,
                                                          conduit_index_t num_elements,
                                                          conduit_index_t offset,
                                                          conduit_index_t stride,
                                                          conduit_index_t element_bytes,
                                                          conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_external_unsigned_short_ptr(conduit_node *cnode,
                                                  unsigned short *data,
                                                  conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_external_unsigned_short_ptr_detailed(conduit_node *cnode,
                                                      unsigned short *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_external_unsigned_int_ptr(conduit_node *cnode,
                                                unsigned int *data,
                                                conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_unsigned_int_ptr_detailed(conduit_node *cnode,
                                                    unsigned int *data,
                                                    conduit_index_t num_elements,
                                                    conduit_index_t offset,
                                                    conduit_index_t stride,
                                                    conduit_index_t element_bytes,
                                                    conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_external_unsigned_long_ptr(conduit_node *cnode,
                                                 unsigned long *data,
                                                 conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_unsigned_long_ptr_detailed(conduit_node *cnode,
                                                     unsigned long *data,
                                                     conduit_index_t num_elements,
                                                     conduit_index_t offset,
                                                     conduit_index_t stride,
                                                     conduit_index_t element_bytes,
                                                     conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}


//-----------------------------------------------------------------------------
// set_external cstyle floating point pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_external_float_ptr(conduit_node *cnode,
                                         float *data,
                                         conduit_index_t num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_float_ptr_detailed(conduit_node *cnode,
                                             float *data,
                                             conduit_index_t num_elements,
                                             conduit_index_t offset,
                                             conduit_index_t stride,
                                             conduit_index_t element_bytes,
                                             conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_double_ptr(conduit_node *cnode,
                                          double *data,
                                          conduit_index_t  num_elements)
{
    cpp_node(cnode)->set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_external_double_ptr_detailed(conduit_node *cnode,
                                              double *data,
                                              conduit_index_t num_elements,
                                              conduit_index_t offset,
                                              conduit_index_t stride,
                                              conduit_index_t element_bytes,
                                              conduit_index_t endianness)
{    
    cpp_node(cnode)->set_external(data,
                                  num_elements,
                                  offset,
                                  stride,
                                  element_bytes,
                                  endianness);
}



//-----------------------------------------------------------------------------
// -- set_path_external via cstyle pointers for (scalar and array types) -- 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// set_path_external cstyle signed integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_char_ptr(conduit_node *cnode,
                                             const char *path,
                                             char *data,
                                             conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external((conduit_byte*)data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_char_ptr_detailed(conduit_node *cnode,
                                                      const char *path,
                                                      char *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external((conduit_byte*)data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_short_ptr(conduit_node *cnode,
                                              const char *path,
                                              short *data,
                                              conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_short_ptr_detailed(conduit_node *cnode,
                                                       const char *path,
                                                       short *data,
                                                       conduit_index_t num_elements,
                                                       conduit_index_t offset,
                                                       conduit_index_t stride,
                                                       conduit_index_t element_bytes,
                                                       conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int_ptr(conduit_node *cnode,
                                            const char *path,
                                            int *data,
                                            conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_int_ptr_detailed(conduit_node *cnode,
                                                     const char *path,
                                                     int *data,
                                                     conduit_index_t num_elements,
                                                     conduit_index_t offset,
                                                     conduit_index_t stride,
                                                     conduit_index_t element_bytes,
                                                     conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_long_ptr(conduit_node *cnode,
                                             const char *path,
                                             long *data,
                                             conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}

//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_long_ptr_detailed(conduit_node *cnode,
                                                      const char *path,
                                                      long *data,
                                                      conduit_index_t num_elements,
                                                      conduit_index_t offset,
                                                      conduit_index_t stride,
                                                      conduit_index_t element_bytes,
                                                      conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}



//-----------------------------------------------------------------------------
// set_path cstyle unsigned integer pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_char_ptr(conduit_node *cnode,
                                                      const char *path,
                                                      unsigned char *data,
                                                      conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_char_ptr_detailed(conduit_node *cnode,
                                                               const char *path,
                                                               unsigned char *data,
                                                               conduit_index_t num_elements,
                                                               conduit_index_t offset,
                                                               conduit_index_t stride,
                                                               conduit_index_t element_bytes,
                                                               conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_short_ptr(conduit_node *cnode,
                                                       const char *path,
                                                       unsigned short *data,
                                                       conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_short_ptr_detailed(conduit_node *cnode,
                                                                const char *path,
                                                                unsigned short *data,
                                                                conduit_index_t num_elements,
                                                                conduit_index_t offset,
                                                                conduit_index_t stride,
                                                                conduit_index_t element_bytes,
                                                                conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_int_ptr(conduit_node *cnode,
                                                     const char *path,
                                                     unsigned int *data,
                                                     conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_int_ptr_detailed(conduit_node *cnode,
                                                              const char *path,
                                                              unsigned int *data,
                                                              conduit_index_t num_elements,
                                                              conduit_index_t offset,
                                                              conduit_index_t stride,
                                                              conduit_index_t element_bytes,
                                                              conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_long_ptr(conduit_node *cnode,
                                                      const char *path,
                                                      unsigned long *data,
                                                      conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_unsigned_long_ptr_detailed(conduit_node *cnode,
                                                               const char *path,
                                                               unsigned long *data,
                                                               conduit_index_t num_elements,
                                                               conduit_index_t offset,
                                                               conduit_index_t stride,
                                                               conduit_index_t element_bytes,
                                                               conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
// set_path cstyle floating point pointer cases
//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_float_ptr(conduit_node *cnode,
                                              const char *path,
                                              float *data,
                                              conduit_index_t num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_float_ptr_detailed(conduit_node *cnode,
                                                       const char *path,
                                                       float *data,
                                                       conduit_index_t num_elements,
                                                       conduit_index_t offset,
                                                       conduit_index_t stride,
                                                       conduit_index_t element_bytes,
                                                       conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_double_ptr(conduit_node *cnode,
                                               const char *path,
                                               double *data,
                                               conduit_index_t  num_elements)
{
    cpp_node(cnode)->fetch(path).set_external(data,num_elements);
}


//-----------------------------------------------------------------------------
void
conduit_node_set_path_external_double_ptr_detailed(conduit_node *cnode,
                                                        const char *path,
                                                        double *data,
                                                        conduit_index_t num_elements,
                                                        conduit_index_t offset,
                                                        conduit_index_t stride,
                                                        conduit_index_t element_bytes,
                                                        conduit_index_t endianness)
{    
    cpp_node(cnode)->fetch(path).set_external(data,
                                              num_elements,
                                              offset,
                                              stride,
                                              element_bytes,
                                              endianness);
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// leaf value access
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// as string access
//-----------------------------------------------------------------------------
char *
conduit_node_as_char8_str(conduit_node *cnode)
{
    return cpp_node(cnode)->as_char8_str();
}

//-----------------------------------------------------------------------------
// direct data pointer access 
//-----------------------------------------------------------------------------
void *
conduit_node_data_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->data_ptr();
}

//-----------------------------------------------------------------------------
// element pointer access 
//-----------------------------------------------------------------------------
void *
conduit_node_element_ptr(conduit_node *cnode,
                         conduit_index_t idx)
{
    return cpp_node(cnode)->element_ptr(idx);
}


//-----------------------------------------------------------------------------
// fetch direct data pointer access 
//-----------------------------------------------------------------------------
void *
conduit_fetch_node_data_ptr(conduit_node *cnode,
                            const char* path)
{
    return cpp_node(cnode)->fetch(path).data_ptr();
}

//-----------------------------------------------------------------------------
// fetch element pointer access 
//-----------------------------------------------------------------------------
void *
conduit_fetch_node_element_ptr(conduit_node *cnode,
                               const char *path,
                               conduit_index_t idx)
{
    return cpp_node(cnode)->fetch(path).element_ptr(idx);
}

//-----------------------------------------------------------------------------
// as bitwidth signed integer scalar types
//-----------------------------------------------------------------------------
conduit_int8
conduit_node_as_int8(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int8();
}
    
//-----------------------------------------------------------------------------
conduit_int16
conduit_node_as_int16(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int16();
}

//-----------------------------------------------------------------------------
conduit_int32
conduit_node_as_int32(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int32();
}


//-----------------------------------------------------------------------------
conduit_int64
conduit_node_as_int64(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int64();
}


//-----------------------------------------------------------------------------
// as bitwidth unsigned integer scalar types
//-----------------------------------------------------------------------------
conduit_uint8
conduit_node_as_uint8(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint8();
}

//-----------------------------------------------------------------------------
conduit_uint16
conduit_node_as_uint16(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint16();
}


//-----------------------------------------------------------------------------
conduit_uint32
conduit_node_as_uint32(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint32();
}


//-----------------------------------------------------------------------------
conduit_uint64
conduit_node_as_uint64(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint64();
}


//-----------------------------------------------------------------------------
// as bitwidth floating point scalar types
//-----------------------------------------------------------------------------
conduit_float32
conduit_node_as_float32(conduit_node *cnode)
{
    return cpp_node(cnode)->as_float32();
}

//-----------------------------------------------------------------------------
conduit_float64
conduit_node_as_float64(conduit_node *cnode)
{
    return cpp_node(cnode)->as_float64();
}


//-----------------------------------------------------------------------------
// as bitwidth signed integer pointer access
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
conduit_int8 *
conduit_node_as_int8_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int8_ptr();
}


//-----------------------------------------------------------------------------
conduit_int16 *
conduit_node_as_int16_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int16_ptr();
}


//-----------------------------------------------------------------------------
conduit_int32 *
conduit_node_as_int32_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int32_ptr();
}


//-----------------------------------------------------------------------------
conduit_int64 *
conduit_node_as_int64_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int64_ptr();
}


//-----------------------------------------------------------------------------
// as bitwidth unsigned integer pointer access
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
conduit_uint8 *
conduit_node_as_uint8_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint8_ptr();
}


//-----------------------------------------------------------------------------
conduit_uint16 *
conduit_node_as_uint16_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint16_ptr();
}


//-----------------------------------------------------------------------------
conduit_uint32 *
conduit_node_as_uint32_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint32_ptr();
}

//-----------------------------------------------------------------------------
conduit_uint64 *
conduit_node_as_uint64_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_uint64_ptr();
}


//-----------------------------------------------------------------------------
// as bitwidth floating point pointer access
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
conduit_float32 *
conduit_node_as_float32_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_float32_ptr();
}


//-----------------------------------------------------------------------------
conduit_float64 *
conduit_node_as_float64_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_float64_ptr();
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// leaf value access via path (bitwidth style types)
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// fetch_path_as bitwidth signed integer scalar types
//-----------------------------------------------------------------------------
conduit_int8
conduit_node_fetch_path_as_int8(conduit_node *cnode,
                                const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int8();
}
    
//-----------------------------------------------------------------------------
conduit_int16
conduit_node_fetch_path_as_int16(conduit_node *cnode,
                                 const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int16();
}

//-----------------------------------------------------------------------------
conduit_int32
conduit_node_fetch_path_as_int32(conduit_node *cnode,
                                 const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int32();
}


//-----------------------------------------------------------------------------
conduit_int64
conduit_node_fetch_path_as_int64(conduit_node *cnode,
                                 const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int64();
}


//-----------------------------------------------------------------------------
// fetch_path_as bitwidth unsigned integer scalar types
//-----------------------------------------------------------------------------
conduit_uint8
conduit_node_fetch_path_as_uint8(conduit_node *cnode,
                                 const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint8();
}

//-----------------------------------------------------------------------------
conduit_uint16
conduit_node_fetch_path_as_uint16(conduit_node *cnode,
                                  const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint16();
}


//-----------------------------------------------------------------------------
conduit_uint32
conduit_node_fetch_path_as_uint32(conduit_node *cnode,
                                  const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint32();
}


//-----------------------------------------------------------------------------
conduit_uint64
conduit_node_fetch_path_as_uint64(conduit_node *cnode,
                                  const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint64();
}


//-----------------------------------------------------------------------------
// fetch_path_as bitwidth floating point scalar types
//-----------------------------------------------------------------------------
conduit_float32
conduit_node_fetch_path_as_float32(conduit_node *cnode,
                                   const char *path)
{
    return cpp_node(cnode)->fetch(path).as_float32();
}

//-----------------------------------------------------------------------------
conduit_float64
conduit_node_fetch_path_as_float64(conduit_node *cnode,
                                   const char *path)
{
    return cpp_node(cnode)->fetch(path).as_float64();
}


//-----------------------------------------------------------------------------
// fetch_path_as bitwidth signed integer pointer access
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
conduit_int8 *
conduit_node_fetch_path_as_int8_ptr(conduit_node *cnode,
                                    const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int8_ptr();
}


//-----------------------------------------------------------------------------
conduit_int16 *
conduit_node_fetch_path_as_int16_ptr(conduit_node *cnode,
                                     const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int16_ptr();
}


//-----------------------------------------------------------------------------
conduit_int32 *
conduit_node_fetch_path_as_int32_ptr(conduit_node *cnode,
                                     const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int32_ptr();
}


//-----------------------------------------------------------------------------
conduit_int64 *
conduit_node_fetch_path_as_int64_ptr(conduit_node *cnode,
                                     const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int64_ptr();
}


//-----------------------------------------------------------------------------
// fetch_path_as bitwidth unsigned integer pointer access
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
conduit_uint8 *
conduit_node_fetch_path_as_uint8_ptr(conduit_node *cnode,
                                      const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint8_ptr();
}


//-----------------------------------------------------------------------------
conduit_uint16 *
conduit_node_fetch_path_as_uint16_ptr(conduit_node *cnode,
                                      const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint16_ptr();
}


//-----------------------------------------------------------------------------
conduit_uint32 *
conduit_node_fetch_path_as_uint32_ptr(conduit_node *cnode,
                                      const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint32_ptr();
}

//-----------------------------------------------------------------------------
conduit_uint64 *
conduit_node_fetch_path_as_uint64_ptr(conduit_node *cnode,
                                      const char *path)
{
    return cpp_node(cnode)->fetch(path).as_uint64_ptr();
}


//-----------------------------------------------------------------------------
// fetch_path_as bitwidth floating point pointer access
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
conduit_float32 *
conduit_node_fetch_path_as_float32_ptr(conduit_node *cnode,
                                       const char *path)
{
    return cpp_node(cnode)->fetch(path).as_float32_ptr();
}


//-----------------------------------------------------------------------------
conduit_float64 *
conduit_node_fetch_path_as_float64_ptr(conduit_node *cnode,
                                  const char *path)
{
    return cpp_node(cnode)->fetch(path).as_float64_ptr();
}



//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// leaf value access (native c style types)
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// as cstyle signed integer scalar access
//-----------------------------------------------------------------------------
char
conduit_node_as_char(conduit_node *cnode)
{
    return cpp_node(cnode)->as_char();
}

//-----------------------------------------------------------------------------
short
conduit_node_as_short(conduit_node *cnode)
{
    return cpp_node(cnode)->as_short();
}

//-----------------------------------------------------------------------------
int
conduit_node_as_int(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int();
}

//-----------------------------------------------------------------------------
long
conduit_node_as_long(conduit_node *cnode)
{
    return cpp_node(cnode)->as_long();
}

//-----------------------------------------------------------------------------
// as cstyle unsigned integer scalar access
//-----------------------------------------------------------------------------
unsigned char
conduit_node_as_unsigned_char(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_char();
}

//-----------------------------------------------------------------------------
unsigned short
conduit_node_as_unsigned_short(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_short();
}

//-----------------------------------------------------------------------------
unsigned int
conduit_node_as_unsigned_int(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_int();
}

//-----------------------------------------------------------------------------
unsigned long
conduit_node_as_unsigned_long(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_long();
}

//-----------------------------------------------------------------------------
// as cstyle floating point scalar access
//-----------------------------------------------------------------------------
float
conduit_node_as_float(conduit_node *cnode)
{
    return cpp_node(cnode)->as_float();
}

//-----------------------------------------------------------------------------
double 
conduit_node_as_double(conduit_node *cnode)
{
    return cpp_node(cnode)->as_double();
}
//-----------------------------------------------------------------------------
// as cstyle signed integer pointer access
//-----------------------------------------------------------------------------
char *
conduit_node_as_char_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_char_ptr();
}

//-----------------------------------------------------------------------------
short *
conduit_node_as_short_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_short_ptr();
}


//-----------------------------------------------------------------------------
int *
conduit_node_as_int_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_int_ptr();
}

//-----------------------------------------------------------------------------
long *
conduit_node_as_long_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_long_ptr();
}


//-----------------------------------------------------------------------------
// as cstyle unsigned integer pointer access
//-----------------------------------------------------------------------------
unsigned char *
conduit_node_as_unsigned_char_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_char_ptr();
}

//-----------------------------------------------------------------------------
unsigned short *
conduit_node_as_unsigned_short_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_short_ptr();
}

//-----------------------------------------------------------------------------
unsigned int *
conduit_node_as_unsigned_int_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_int_ptr();
}

//-----------------------------------------------------------------------------
unsigned long *
conduit_node_as_unsigned_long_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_unsigned_long_ptr();
}


//-----------------------------------------------------------------------------
// as cstyle floating point pointer access
//-----------------------------------------------------------------------------
float *
conduit_node_as_float_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_float_ptr();
}
    
double *
conduit_node_as_double_ptr(conduit_node *cnode)
{
    return cpp_node(cnode)->as_double_ptr();
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// leaf value access via path (native c style types)
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// fetch_path_as cstyle signed integer scalar access
//-----------------------------------------------------------------------------
char
conduit_node_fetch_path_as_char(conduit_node *cnode,
                                const char *path)
{
    return cpp_node(cnode)->fetch(path).as_char();
}

//-----------------------------------------------------------------------------
short
conduit_node_fetch_path_as_short(conduit_node *cnode,
                                 const char *path)
{
    return cpp_node(cnode)->fetch(path).as_short();
}

//-----------------------------------------------------------------------------
int
conduit_node_fetch_path_as_int(conduit_node *cnode,
                               const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int();
}

//-----------------------------------------------------------------------------
long
conduit_node_fetch_path_as_long(conduit_node *cnode,
                                const char *path)
{
    return cpp_node(cnode)->fetch(path).as_long();
}

//-----------------------------------------------------------------------------
// fetch_path_as cstyle unsigned integer scalar access
//-----------------------------------------------------------------------------
unsigned char
conduit_node_fetch_path_as_unsigned_char(conduit_node *cnode,
                                         const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_char();
}


//-----------------------------------------------------------------------------
unsigned short
conduit_node_fetch_path_as_unsigned_short(conduit_node *cnode,
                                          const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_short();
}

//-----------------------------------------------------------------------------
unsigned int
conduit_node_fetch_path_as_unsigned_int(conduit_node *cnode,
                                        const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_int();
}


//-----------------------------------------------------------------------------
unsigned long
conduit_node_fetch_path_as_unsigned_long(conduit_node *cnode,
                                         const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_long();
}

//-------------------------------------------------------------------------
// fetch_path_as cstyle floating point scalar access
//-------------------------------------------------------------------------
float
conduit_node_fetch_path_as_float(conduit_node *cnode,
                                const char *path)
{
    return cpp_node(cnode)->fetch(path).as_float();
}

//-------------------------------------------------------------------------
double
conduit_node_fetch_path_as_double(conduit_node *cnode,
                                  const char *path)
{
    return cpp_node(cnode)->fetch(path).as_double();
}

//-------------------------------------------------------------------------
// fetch_path_as cstyle signed integer pointer access
//-------------------------------------------------------------------------
char *
conduit_node_fetch_path_as_char_ptr(conduit_node *cnode,
                                    const char *path)
{
    return cpp_node(cnode)->fetch(path).as_char_ptr();
}

//-------------------------------------------------------------------------
short *
conduit_node_fetch_path_as_short_ptr(conduit_node *cnode,
                                     const char *path)
{
    return cpp_node(cnode)->fetch(path).as_short_ptr();
}

//-------------------------------------------------------------------------
int *
conduit_node_fetch_path_as_int_ptr(conduit_node *cnode,
                                   const char *path)
{
    return cpp_node(cnode)->fetch(path).as_int_ptr();
}


//-------------------------------------------------------------------------
long *
conduit_node_fetch_path_as_long_ptr(conduit_node *cnode,
                                    const char *path)
{
    return cpp_node(cnode)->fetch(path).as_long_ptr();
}

//-------------------------------------------------------------------------
// fetch_path_as cstyle unsigned integer pointer access
//-------------------------------------------------------------------------
unsigned char *
conduit_node_fetch_path_as_unsigned_char_ptr(conduit_node *cnode,
                                             const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_char_ptr();
}

//-------------------------------------------------------------------------
unsigned short *
conduit_node_fetch_path_as_unsigned_short_ptr(conduit_node *cnode,
                                              const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_short_ptr();
}


//-------------------------------------------------------------------------
unsigned int *
conduit_node_fetch_path_as_unsigned_int_ptr(conduit_node *cnode,
                                            const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_int_ptr();
}

//-------------------------------------------------------------------------
unsigned long *
conduit_node_fetch_path_as_unsigned_long_ptr(conduit_node *cnode,
                                             const char *path)
{
    return cpp_node(cnode)->fetch(path).as_unsigned_long_ptr();
}

//-------------------------------------------------------------------------
// fetch_path_as cstyle floating point pointer access
//-------------------------------------------------------------------------
float *
conduit_node_fetch_path_as_float_ptr(conduit_node *cnode,
                                     const char *path)
{
    return cpp_node(cnode)->fetch(path).as_float_ptr();
}


//-------------------------------------------------------------------------
double *
conduit_node_fetch_path_as_double_ptr(conduit_node *cnode,
                                      const char *path)
{
    return cpp_node(cnode)->fetch(path).as_double_ptr();
}

//-----------------------------------------------------------------------------
// fetch_path_as string pointer access
//-----------------------------------------------------------------------------
char *
conduit_node_fetch_path_as_char8_str(conduit_node *cnode,
                                     const char *path)
{
    return cpp_node(cnode)->fetch(path).as_char8_str();
}


}
//-----------------------------------------------------------------------------
// -- end extern C
//-----------------------------------------------------------------------------

