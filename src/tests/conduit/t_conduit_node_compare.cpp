//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
/// file: t_conduit_node_compare.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

// test cases:
// - verify that a node always diffs properly against itself
// - verify that nodes with different data types are always diff
// - verify that a number of leaf node types work (esp. arrays)
// - verify for the object case
//   - no difference
//   - only extra items in the self
//   - only extra items in the other
//   - different leaf values
// - verify for the list case
//   - no difference
//   - only difference is size
//   - different in a few elements, which show up properly

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, basic)
{
    Node n, info;
    EXPECT_FALSE(n.diff(n, info));
    EXPECT_TRUE(info.dtype().is_empty());

    n.set("");
    EXPECT_FALSE(n.diff(n, info));
    EXPECT_TRUE(info.dtype().is_empty());

    Node o;
    n.set(1);
    o.set(2);
    EXPECT_TRUE(n.diff(o, info));
    EXPECT_FALSE(info.dtype().is_empty());
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, leaf_numeric)
{
    DataType::TypeID leaf_tids[10] = {
        DataType::INT8_ID, DataType::INT16_ID, DataType::INT32_ID, DataType::INT64_ID,
        DataType::UINT8_ID, DataType::UINT16_ID, DataType::UINT32_ID, DataType::UINT64_ID,
        DataType::FLOAT32_ID, DataType::FLOAT64_ID
    };

    for(size_t leaf_idx = 0; leaf_idx < 1; leaf_idx++)
    {
        DataType::TypeID leaf_tid = leaf_tids[leaf_idx];
        DataType leaf_type_tmp(leaf_tid, 1, 0, 0, 0, Endianness::DEFAULT_ID);
        DataType leaf_type(leaf_tid, 5, 0, leaf_type_tmp.bytes_compact(),
            0, Endianness::DEFAULT_ID);

        const size_t type_bytes = leaf_type_tmp.bytes_compact();
        const size_t leaf_bytes = 5 * type_bytes;

        conduit_byte* n_data = new conduit_byte[leaf_bytes];
        if(leaf_tid == DataType::INT8_ID)         *((int8*)n_data) = 0;
        else if(leaf_tid == DataType::INT16_ID)   *((int16*)n_data) = 0;
        else if(leaf_tid == DataType::INT32_ID)   *((int32*)n_data) = 0;
        else if(leaf_tid == DataType::INT64_ID)   *((int64*)n_data) = 0;
        else if(leaf_tid == DataType::UINT8_ID)   *((uint8*)n_data) = 0;
        else if(leaf_tid == DataType::UINT16_ID)  *((uint16*)n_data) = 0;
        else if(leaf_tid == DataType::UINT32_ID)  *((uint32*)n_data) = 0;
        else if(leaf_tid == DataType::UINT64_ID)  *((uint64*)n_data) = 0;
        else if(leaf_tid == DataType::FLOAT32_ID) *((float32*)n_data) = 0.0f;
        else if(leaf_tid == DataType::FLOAT64_ID) *((float64*)n_data) = 0.0;
        Node n(leaf_type, (void*)n_data, true);

        conduit_byte* o_data = new conduit_byte[leaf_bytes];
        memcpy(o_data, n_data, leaf_bytes);
        Node o(leaf_type, (void*)o_data, true);

        Node info;
        EXPECT_FALSE(n.diff(o, info));
        EXPECT_TRUE(info.dtype().is_empty());

        // TODO(JRC): There's something wrong with how the fourth element
        // of the list is being set here that needs to be fixed.
        info.reset();
        memset(&o_data[0*type_bytes], 1, 1);
        memset(&o_data[4*type_bytes], 1, 1);
        EXPECT_TRUE(n.diff(o, info));
        EXPECT_FALSE(info.dtype().is_empty());

        EXPECT_EQ(info.number_of_children(), 2);
        EXPECT_EQ(info.child(0).as_string()[0], '0');
        EXPECT_EQ(info.child(1).as_string()[0], '4');

        delete [] n_data;
        delete [] o_data;
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, leaf_string)
{
    // TODO(JRC): Add a test case here.
    EXPECT_TRUE(true);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, leaf_mismatch)
{
    // TODO(JRC): Add a test case here.
    EXPECT_TRUE(true);
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, test)
{
    // TODO(JRC): Add a test case here.
    EXPECT_TRUE(true);
}
