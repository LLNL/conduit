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
/// file: t_relay_zfp.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_zfp.hpp"
#include "gtest/gtest.h"
#include <cstring>

using namespace conduit;
using namespace conduit::relay;

//-----------------------------------------------------------------------------
TEST(conduit_relay_zfp, zfp_read_and_verify_header)
{
    // initialize empty result Node
    Node result;
    EXPECT_FALSE(result.has_child(io::ZFP_HEADER_FIELD_NAME));

    // create compressed-array
    uint nx = 9;
    uint ny = 12;
    double rate = 8.0;
    zfp::array2f arr(nx, ny, rate);

    // write zfparray to Node
    EXPECT_EQ(0, io::zfp_read(&arr, result));

    // verify header entry was set
    EXPECT_TRUE(result.has_child(io::ZFP_HEADER_FIELD_NAME));
    Node n_header = result[io::ZFP_HEADER_FIELD_NAME];

    // assert header dtype
    EXPECT_TRUE(n_header.dtype().is_uint8());
    uint8_array header_as_arr = n_header.as_uint8_array();

    // assert header length
    zfp::array::header header = arr.get_header();
    EXPECT_EQ(sizeof(header), header_as_arr.number_of_elements());

    // assert header contents
    EXPECT_TRUE(0 == std::memcmp(header.buffer, n_header.data_ptr(), sizeof(header)));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_zfp, zfp_read_and_verify_compressed_data)
{
    // initialize empty result Node
    Node result;
    EXPECT_FALSE(result.has_child(io::ZFP_COMPRESSED_DATA_FIELD_NAME));

    // create compressed-array
    uint nx = 9;
    uint ny = 12;

    float vals[nx * ny] = {0};
    uint i;
    for (i = 0; i < nx*ny; i++) {
        vals[i] = i*i;
    }

    double rate = 8.0;
    zfp::array2f arr(nx, ny, rate, vals);

    // write zfparray to Node
    EXPECT_EQ(0, io::zfp_read(&arr, result));

    // verify compressed data entry was set
    EXPECT_TRUE(result.has_child(io::ZFP_COMPRESSED_DATA_FIELD_NAME));
    Node n_data = result[io::ZFP_COMPRESSED_DATA_FIELD_NAME];

    EXPECT_TRUE(n_data.dtype().is_unsigned_integer());
    size_t compressed_data_num_words = arr.compressed_size() * CHAR_BIT / stream_word_bits;

    // compressed-data entry written with same uint type, having width `stream_word_bits`
    switch(stream_word_bits) {
        case 64:
            EXPECT_TRUE(n_data.dtype().is_uint64());
            {
                uint64_array data_as_arr = n_data.as_uint64_array();
                EXPECT_EQ(compressed_data_num_words, data_as_arr.number_of_elements());
            }
            break;

        case 32:
            EXPECT_TRUE(n_data.dtype().is_uint32());
            {
                uint32_array data_as_arr = n_data.as_uint32_array();
                EXPECT_EQ(compressed_data_num_words, data_as_arr.number_of_elements());
            }
            break;

        case 16:
            EXPECT_TRUE(n_data.dtype().is_uint16());
            {
                uint16_array data_as_arr = n_data.as_uint16_array();
                EXPECT_EQ(compressed_data_num_words, data_as_arr.number_of_elements());
            }
            break;

        case 8:
            EXPECT_TRUE(n_data.dtype().is_uint8());
            {
                uint8_array data_as_arr = n_data.as_uint8_array();
                EXPECT_EQ(compressed_data_num_words, data_as_arr.number_of_elements());
            }
            break;

        default:
            FAIL() << "ZFP was compiled with an unrecognizable word type";
            break;
    }

    EXPECT_TRUE(0 == std::memcmp(arr.compressed_data(), n_data.data_ptr(), arr.compressed_size()));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_zfp, zfp_read_with_header_exception)
{
    // create compressed-array that does not support short header
    uint nx = 9;
    uint ny = 12;
    uint nz = 5;
    double rate = 64.0;
    zfp::array3d arr(nx, ny, nz, rate);

    // write zfparray to Node, but expect failure
    Node result;
    EXPECT_EQ(1, io::zfp_read(&arr, result));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_zfp, zfp_write)
{
    // create compressed-array
    uint nx = 9;
    uint ny = 12;

    float vals[nx * ny];
    uint i, j;
    for (j = 0; j < ny; j++) {
        for (i = 0; i < nx; i++) {
            vals[nx*j + i] = i * 10. + j*j;
        }
    }

    double rate = 32.0;
    zfp::array2f original_arr(nx, ny, rate, vals);

    // write zfparray to Node
    Node result;
    EXPECT_EQ(0, io::zfp_read(&original_arr, result));

    // fetch zfparray object from Node
    zfp::array* fetched_arr = io::zfp_write(result);

    // verify against original_arr
    ASSERT_TRUE(fetched_arr != 0);

    zfp::array2f* casted_arr = dynamic_cast<zfp::array2f*>(fetched_arr);
    ASSERT_TRUE(casted_arr != 0);

    EXPECT_EQ(nx, casted_arr->size_x());
    EXPECT_EQ(ny, casted_arr->size_y());
    EXPECT_EQ(rate, casted_arr->rate());

    // verify compressed data
    EXPECT_EQ(original_arr.compressed_size(), casted_arr->compressed_size());
    EXPECT_TRUE(0 == std::memcmp(original_arr.compressed_data(), casted_arr->compressed_data(), original_arr.compressed_size()));

    delete fetched_arr;
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_zfp, zfp_write_with_exception)
{
    // create compressed-array
    uint nx = 9;
    uint ny = 12;

    float vals[nx * ny] = {0};
    double rate = 32.0;
    zfp::array2f original_arr(nx, ny, rate);

    // write zfparray to Node
    Node result;
    EXPECT_EQ(0, io::zfp_read(&original_arr, result));

    // corrupt the Node's data
    result[io::ZFP_HEADER_FIELD_NAME].set(vals, sizeof(vals));

    // fetch zfparray object from Node
    zfp::array* fetched_arr = io::zfp_write(result);

    // verify no instance returned
    ASSERT_TRUE(fetched_arr == 0);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_zfp, zfp_write_with_compressed_data_dtype_mismatched_with_compiled_zfp_word_size)
{
    // create compressed-array
    uint nx = 9;
    uint ny = 12;
    uint nz = 15;

    double rate = 16.0;
    zfp::array3f arr(nx, ny, nz, rate);

    // write zfparray to Node
    Node result;
    EXPECT_EQ(0, relay::io::zfp_read(&arr, result));

    // remove compressed-data node
    EXPECT_TRUE(result.has_child(io::ZFP_COMPRESSED_DATA_FIELD_NAME));
    result.remove(io::ZFP_COMPRESSED_DATA_FIELD_NAME);

    // re-add compressed-data node as the wrong type
    switch(stream_word_bits) {
        case 64:
        case 32:
        case 16:
            {
                uint8 data = 3;
                result[io::ZFP_COMPRESSED_DATA_FIELD_NAME] = data;
            }
            break;

        case 8:
            {
                uint64 data = 3;
                result[io::ZFP_COMPRESSED_DATA_FIELD_NAME] = data;
            }
            break;

        default:
            FAIL() << "ZFP was compiled with an unrecognizable word type";
    }

    // fetch zfparray object from Node
    zfp::array* fetched_arr = io::zfp_write(result);

    // verify no instance returned
    ASSERT_TRUE(fetched_arr == 0);
}

