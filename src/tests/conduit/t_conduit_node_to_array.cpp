//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_node_to_array.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

// TODO(JRC): Figure out if there's any better way to facilitate these conversions
// from C++ object functions to C-Style functions for better procedural testing.

typedef void (*NodeConvertFun)(const Node&, Node&);

void convert_to_int8_array(const Node &n, Node &res) { n.to_int8_array(res); }
void convert_to_int16_array(const Node &n, Node &res) { n.to_int16_array(res); }
void convert_to_int32_array(const Node &n, Node &res) { n.to_int32_array(res); }
void convert_to_int64_array(const Node &n, Node &res) { n.to_int64_array(res); }
void convert_to_uint8_array(const Node &n, Node &res) { n.to_uint8_array(res); }
void convert_to_uint16_array(const Node &n, Node &res) { n.to_uint16_array(res); }
void convert_to_uint32_array(const Node &n, Node &res) { n.to_uint32_array(res); }
void convert_to_uint64_array(const Node &n, Node &res) { n.to_uint64_array(res); }
void convert_to_float32_array(const Node &n, Node &res) { n.to_float32_array(res); }
void convert_to_float64_array(const Node &n, Node &res) { n.to_float64_array(res); }
void convert_to_char_array(const Node &n, Node &res) { n.to_char_array(res); }
void convert_to_short_array(const Node &n, Node &res) { n.to_short_array(res); }
void convert_to_int_array(const Node &n, Node &res) { n.to_int_array(res); }
void convert_to_long_array(const Node &n, Node &res) { n.to_long_array(res); }
void convert_to_unsigned_char_array(const Node &n, Node &res) { n.to_unsigned_char_array(res); }
void convert_to_unsigned_short_array(const Node &n, Node &res) { n.to_unsigned_short_array(res); }
void convert_to_unsigned_int_array(const Node &n, Node &res) { n.to_unsigned_int_array(res); }
void convert_to_unsigned_long_array(const Node &n, Node &res) { n.to_unsigned_long_array(res); }
void convert_to_float_array(const Node &n, Node &res) { n.to_float_array(res); }
void convert_to_double_array(const Node &n, Node &res) { n.to_double_array(res); }

const int CONVERT_TYPES[20] = {
    DataType::INT8_ID, DataType::INT16_ID, DataType::INT32_ID, DataType::INT64_ID,
    DataType::UINT8_ID, DataType::UINT16_ID, DataType::UINT32_ID, DataType::UINT64_ID,
    DataType::FLOAT32_ID, DataType::FLOAT64_ID,
    CONDUIT_NATIVE_CHAR_ID, CONDUIT_NATIVE_SHORT_ID, CONDUIT_NATIVE_INT_ID, CONDUIT_NATIVE_LONG_ID,
    CONDUIT_NATIVE_UNSIGNED_CHAR_ID, CONDUIT_NATIVE_UNSIGNED_SHORT_ID, CONDUIT_NATIVE_UNSIGNED_INT_ID, CONDUIT_NATIVE_UNSIGNED_LONG_ID,
    CONDUIT_NATIVE_FLOAT_ID, CONDUIT_NATIVE_DOUBLE_ID
};

const NodeConvertFun CONVERT_TO_FUNS[20] = {
    convert_to_int8_array, convert_to_int16_array, convert_to_int32_array, convert_to_int64_array,
    convert_to_uint8_array, convert_to_uint16_array, convert_to_uint32_array, convert_to_uint64_array,
    convert_to_float32_array, convert_to_float64_array,
    convert_to_char_array, convert_to_short_array, convert_to_int_array, convert_to_long_array,
    convert_to_unsigned_char_array, convert_to_unsigned_short_array, convert_to_unsigned_int_array, convert_to_unsigned_long_array,
    convert_to_float_array, convert_to_double_array
};

//-----------------------------------------------------------------------------
TEST(conduit_node_to_array, static_type_to_vector)
{
    uint8 data_vals[3] = {1,2,3};
    Node data_node;
    data_node.set(data_vals,3);

    for(index_t ti = 0; ti < 20; ti++)
    {
        int node_tid = CONVERT_TYPES[ti];
        NodeConvertFun convert_fun = CONVERT_TO_FUNS[ti];

        Node type_node;
        convert_fun(data_node, type_node);

        EXPECT_EQ(type_node.dtype().id(), node_tid);
        EXPECT_EQ(type_node.dtype().number_of_elements(),
            data_node.dtype().number_of_elements());

        Node temp_node;
        convert_to_uint8_array(type_node, temp_node);
        EXPECT_FALSE(memcmp(temp_node.data_ptr(), data_node.data_ptr(),
            data_node.total_bytes_allocated()));
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_to_array, dynamic_type_to_vector)
{
    uint8 data_vals[3] = {1,2,3};
    Node data_node;
    data_node.set(data_vals,3);

    for(index_t ti = 0; ti < 10; ti++)
    {
        Node base_node;
        const DataType::TypeID base_tid = (DataType::TypeID)CONVERT_TYPES[ti];
        NodeConvertFun base_convert_fun = CONVERT_TO_FUNS[ti];
        base_convert_fun(data_node, base_node);

        for(index_t tj = 0; tj < 10; tj++)
        {
            Node to_node;
            const DataType::TypeID to_tid = (DataType::TypeID)CONVERT_TYPES[tj];
            base_node.to_data_type(to_tid, to_node);

            EXPECT_EQ(to_node.dtype().id(), to_tid);
            EXPECT_EQ(to_node.dtype().number_of_elements(),
                data_node.dtype().number_of_elements());

            Node temp_node;
            convert_to_uint8_array(to_node, temp_node);
            EXPECT_FALSE(memcmp(temp_node.data_ptr(), data_node.data_ptr(),
                data_node.total_bytes_allocated()));
        }
    }
}
