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
#include "conduit_utils.hpp"

#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;

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
    const DataType::TypeID leaf_tids[10] = {
        DataType::INT8_ID, DataType::INT16_ID, DataType::INT32_ID, DataType::INT64_ID,
        DataType::UINT8_ID, DataType::UINT16_ID, DataType::UINT32_ID, DataType::UINT64_ID,
        DataType::FLOAT32_ID, DataType::FLOAT64_ID
    };

    for(size_t type_idx = 0; type_idx < 10; type_idx++)
    {
        const DataType::TypeID leaf_tid = leaf_tids[type_idx];
        DataType leaf_type(leaf_tid, 5);

        const size_t type_bytes = leaf_type.stride();
        const size_t leaf_bytes = leaf_type.bytes_compact();

        conduit_byte* n_data = new conduit_byte[leaf_bytes];
        memset(n_data, 0, leaf_bytes);
        Node n(leaf_type, (void*)n_data, true);

        conduit_byte* o_data = new conduit_byte[leaf_bytes];
        memcpy(o_data, n_data, leaf_bytes);
        Node o(leaf_type, (void*)o_data, true);

        Node info;
        EXPECT_FALSE(n.diff(o, info));
        EXPECT_TRUE(info.dtype().is_empty());

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
    const std::string test_strs[4] = {"I", "me", "You", "tHeM"};

    for(size_t test_idx = 0; test_idx < 4; test_idx++)
    {
        const std::string test_str = test_strs[test_idx];
        std::string diff_str = test_str;
        {
            size_t test_len = test_str.length();
            diff_str[test_len-1] += 1;
        }

        Node n, o, info;
        n.set(test_str);
        o.set(test_str);
        EXPECT_FALSE(n.diff(o, info));
        EXPECT_TRUE(info.dtype().is_empty());

        info.reset();
        n.set(test_str);
        o.set(diff_str);
        EXPECT_TRUE(n.diff(o, info));
        EXPECT_FALSE(info.dtype().is_empty());

        const std::string info_str = info.as_string();
        EXPECT_NE(info_str.find(test_str), std::string::npos);
        EXPECT_NE(info_str.find(diff_str), std::string::npos);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, leaf_mismatch)
{
    const DataType::TypeID leaf_tids[6] = {
        DataType::INT32_ID, DataType::INT64_ID,
        DataType::UINT32_ID, DataType::UINT64_ID,
        DataType::FLOAT32_ID, DataType::FLOAT64_ID
    };

    for(size_t type_idx = 0; type_idx < 6; type_idx++)
    {
        const DataType::TypeID curr_tid = leaf_tids[(type_idx+0)%6];
        const DataType::TypeID next_tid = leaf_tids[(type_idx+1)%6];

        DataType curr_type(curr_tid, 1);
        DataType next_type(next_tid, 1);

        const size_t max_bytes = std::max(
            curr_type.bytes_compact(), next_type.bytes_compact());
        conduit_byte* max_data = new conduit_byte[max_bytes];
        memset(max_data, 0, max_bytes);

        Node n(curr_type, (void*)max_data, true);
        Node o(next_type, (void*)max_data, true);

        Node info;
        EXPECT_TRUE(n.diff(o, info));
        EXPECT_FALSE(info.dtype().is_empty());

        delete [] max_data;
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, object_item_diff)
{
    const size_t n_num_children = 5;
    Node n, o, info;
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        std::ostringstream oss;
        oss << leaf_idx;
        n[oss.str()].set(leaf_idx);
        o[oss.str()].set(leaf_idx+(leaf_idx%2));
    }

    EXPECT_TRUE(n.diff(o, info));
    EXPECT_FALSE(info.dtype().is_empty());

    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        std::ostringstream oss;
        oss << leaf_idx;
        std::string leaf_str = oss.str();
        if(leaf_idx % 2 == 1)
        {
            EXPECT_TRUE(info.has_child(leaf_str));
            EXPECT_TRUE(info[leaf_str].dtype().is_list());
        }
        else
        {
            EXPECT_FALSE(info.has_child(leaf_str));
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, object_size_diff)
{
    const size_t n_num_children = 5;
    Node n, info;
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        std::ostringstream oss;
        oss << leaf_idx;
        n[oss.str()].set(leaf_idx);
    }

    // Full vs. Empty Node Check //

    Node o(DataType::object());
    info.reset();
    EXPECT_TRUE(n.diff(o, info));
    EXPECT_FALSE(info.dtype().is_empty());
    EXPECT_EQ(info.number_of_children(), n_num_children);
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        std::ostringstream oss;
        oss << leaf_idx;
        std::string leaf_str = oss.str();

        EXPECT_TRUE(info.has_child(leaf_str));
        EXPECT_TRUE(info[leaf_str].dtype().is_string());
        EXPECT_NE(info[leaf_str].as_string().find("subtree"), std::string::npos);
    }

    // Equal Node Check //

    o.set(n);
    info.reset();
    EXPECT_FALSE(n.diff(o, info));
    EXPECT_TRUE(info.dtype().is_empty());

    // Half-Full Node Check //

    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        if(leaf_idx % 2 == 1)
        {
            std::ostringstream oss;
            oss << leaf_idx;
            o.remove(oss.str());
        }
    }
    info.reset();
    EXPECT_TRUE(n.diff(o, info));
    EXPECT_FALSE(info.dtype().is_empty());
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        std::ostringstream oss;
        oss << leaf_idx;
        std::string leaf_str = oss.str();
        if(leaf_idx % 2 == 1)
        {
            EXPECT_TRUE(info.has_child(leaf_str));
            EXPECT_TRUE(info[leaf_str].dtype().is_string());
            EXPECT_NE(info[leaf_str].as_string().find("subtree"), std::string::npos);
        }
        else
        {
            EXPECT_FALSE(info.has_child(leaf_str));
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, list_item_diff)
{
    const size_t n_num_children = 5;
    Node n, o, info;
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        n.append().set(leaf_idx);
        o.append().set(leaf_idx+(leaf_idx%2));
    }

    EXPECT_TRUE(n.diff(o, info));
    EXPECT_FALSE(info.dtype().is_empty());
    EXPECT_EQ(info.number_of_children(), n_num_children);

    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        if(leaf_idx % 2 == 1)
        {
            EXPECT_TRUE(info.child(leaf_idx).dtype().is_list());
        }
        else
        {
            EXPECT_TRUE(info.child(leaf_idx).dtype().is_empty());
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, list_size_diff)
{
    const size_t n_num_children = 5;
    Node n, info;
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        n.append().set(leaf_idx);
    }

    // Full vs. Empty Node Check //

    Node o(DataType::list());
    info.reset();
    EXPECT_TRUE(n.diff(o, info));
    EXPECT_FALSE(info.dtype().is_empty());
    EXPECT_EQ(info.number_of_children(), n_num_children);
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        EXPECT_TRUE(info[leaf_idx].dtype().is_string());
        EXPECT_NE(info[leaf_idx].as_string().find("item"), std::string::npos);
    }

    // Equal Node Check //

    o.set(n);
    info.reset();
    EXPECT_FALSE(n.diff(o, info));
    EXPECT_TRUE(info.dtype().is_empty());

    // Half-Full Node Check //

    for(int32 leaf_idx = n_num_children - 1; leaf_idx > n_num_children / 2; leaf_idx--)
    {
        o.remove(leaf_idx);
    }
    info.reset();
    EXPECT_TRUE(n.diff(o, info));
    EXPECT_FALSE(info.dtype().is_empty());
    for(int32 leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        if(leaf_idx > n_num_children / 2)
        {
            EXPECT_TRUE(info[leaf_idx].dtype().is_string());
            EXPECT_NE(info[leaf_idx].as_string().find("item"), std::string::npos);
        }
        else
        {
            EXPECT_TRUE(info[leaf_idx].dtype().is_empty());
        }
    }
}
