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

// TODO(JRC): Revise all of the more intricate test cases below once the format
// for the 'info' node returned by 'Node::equals' is finalized.

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_basic)
{
    Node n, info;
    EXPECT_TRUE(n.equals(n, info));

    n.set("");
    EXPECT_TRUE(n.equals(n, info));

    Node o;
    n.set(1);
    o.set(2);
    EXPECT_FALSE(n.equals(o, info));
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_leaf_numeric)
{
    const DataType::TypeID leaf_tids[10] = {
        DataType::INT8_ID, DataType::INT16_ID, DataType::INT32_ID, DataType::INT64_ID,
        DataType::UINT8_ID, DataType::UINT16_ID, DataType::UINT32_ID, DataType::UINT64_ID,
        DataType::FLOAT32_ID, DataType::FLOAT64_ID
    };

    for(index_t type_idx = 0; type_idx < 10; type_idx++)
    {
        const DataType::TypeID leaf_tid = leaf_tids[type_idx];
        DataType leaf_type(leaf_tid, 5);

        const index_t type_bytes = leaf_type.stride();
        const index_t leaf_bytes = leaf_type.bytes_compact();

        conduit_byte* n_data = new conduit_byte[leaf_bytes];
        memset(n_data, 0, leaf_bytes);
        Node n(leaf_type, (void*)n_data, true);

        conduit_byte* o_data = new conduit_byte[leaf_bytes];
        memcpy(o_data, n_data, leaf_bytes);
        Node o(leaf_type, (void*)o_data, true);

        Node info;
        EXPECT_TRUE(n.equals(o, info, 0.0f));

        info.reset();
        memset(&o_data[0*type_bytes], 1, 1);
        memset(&o_data[4*type_bytes], 1, 1);
        EXPECT_FALSE(n.equals(o, info, 0.0f));

        Node &info_diff = info.child(1);

        EXPECT_EQ(info_diff.dtype().id(), leaf_tid);
        EXPECT_EQ(info_diff.dtype().number_of_elements(), 5);
        for(index_t val_idx = 0; val_idx < 5; val_idx++)
        {
            bool should_uneq = val_idx == 0 || val_idx == 4;
            bool are_uneq = memcmp(&n_data[val_idx*type_bytes],
                                   &o_data[val_idx*type_bytes], type_bytes);
            EXPECT_EQ(are_uneq, should_uneq);
        }

        delete [] n_data;
        delete [] o_data;
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_leaf_string)
{
    const std::string test_strs[4] = {"I", "me", "You", "tHeM"};

    for(index_t test_idx = 0; test_idx < 4; test_idx++)
    {
        const std::string test_str = test_strs[test_idx];
        std::string diff_str = test_str;
        {
            index_t test_len = test_str.length();
            diff_str[test_len-1] += 1;
        }

        Node n, o, info;
        n.set(test_str);
        o.set(test_str);
        EXPECT_TRUE(n.equals(o, info));

        info.reset();
        n.set(test_str);
        o.set(diff_str);
        EXPECT_FALSE(n.equals(o, info));
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_leaf_mismatch)
{
    const DataType::TypeID leaf_tids[6] = {
        DataType::INT32_ID, DataType::INT64_ID,
        DataType::UINT32_ID, DataType::UINT64_ID,
        DataType::FLOAT32_ID, DataType::FLOAT64_ID
    };

    for(index_t type_idx = 0; type_idx < 6; type_idx++)
    {
        const DataType::TypeID curr_tid = leaf_tids[(type_idx+0)%6];
        const DataType::TypeID next_tid = leaf_tids[(type_idx+1)%6];

        DataType curr_type(curr_tid, 1);
        DataType next_type(next_tid, 1);

        const index_t max_bytes = std::max(
            curr_type.bytes_compact(), next_type.bytes_compact());
        conduit_byte* max_data = new conduit_byte[max_bytes];
        memset(max_data, 0, max_bytes);

        Node n(curr_type, (void*)max_data, true);
        Node o(next_type, (void*)max_data, true);

        Node info;
        EXPECT_FALSE(n.equals(o, info));

        delete [] max_data;
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_object_item_diff)
{
    const index_t n_num_children = 5;
    Node n, o, info;
    for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
    {
        std::ostringstream oss;
        oss << child_idx;
        n[oss.str()].set(child_idx);
        o[oss.str()].set(child_idx+(child_idx%2));
    }

    EXPECT_FALSE(n.equals(o, info));

    Node &info_this = info.child(0), &info_children = info.child(1);
    for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
    {
        std::ostringstream oss;
        oss << child_idx;
        std::string child_str = oss.str();

        EXPECT_TRUE(info_children.has_child(child_str));
        EXPECT_EQ(
            info_children.fetch(child_str).child(0).fetch("valid").as_string(),
            (child_idx % 2 == 0) ? "true" : "false");
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_object_size_diff)
{
    const index_t n_num_children = 5;
    Node n;
    for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
    {
        std::ostringstream oss;
        oss << child_idx;
        n[oss.str()].set(child_idx);
    }

    { // Full vs. Empty Node Check //
        Node o(DataType::object()), info;
        EXPECT_FALSE(n.equals(o, info));

        Node &info_this = info.child(0), &info_children = info.child(1);
        EXPECT_TRUE(info_this.has_child("errors"));
        EXPECT_EQ(info_children.number_of_children(), 0);

        Node &info_errors = info_this.fetch("errors");
        EXPECT_EQ(info_errors.number_of_children(), n_num_children);
        for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
        {
            Node &info_child = info_errors.child(child_idx);
            EXPECT_TRUE(info_child.dtype().is_string());
            EXPECT_NE(info_child.as_string().find("arg"), std::string::npos);
        }
    }

    { // Equal Node Check //
        Node o(n), info;
        EXPECT_TRUE(n.equals(o, info));

        Node &info_this = info.child(0), &info_children = info.child(1);
        EXPECT_FALSE(info_this.has_child("errors"));
        EXPECT_EQ(info_children.number_of_children(), n_num_children);

        for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
        {
            std::ostringstream oss;
            oss << child_idx;
            std::string child_str = oss.str();
            EXPECT_TRUE(info_children.has_child(child_str));
        }
    }

    { // Half-Full Node Check //
        Node o(n), info;
        for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
        {
            if(child_idx % 2 == 1)
            {
                std::ostringstream oss;
                oss << child_idx;
                o.remove(oss.str());
            }
        }

        EXPECT_FALSE(n.equals(o, info));

        Node &info_this = info.child(0), &info_children = info.child(1);
        EXPECT_TRUE(info_this.has_child("errors"));
        EXPECT_EQ(info_this.fetch("errors").number_of_children(), n_num_children/2);

        for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
        {
            std::ostringstream oss;
            oss << child_idx;
            std::string child_str = oss.str();

            if(child_idx % 2 != 1)
            {
                EXPECT_TRUE(info_children.has_child(child_str));
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_list_item_diff)
{
    const index_t n_num_children = 5;
    Node n, o, info;
    for(index_t leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        n.append().set(leaf_idx);
        o.append().set(leaf_idx+(leaf_idx%2));
    }

    EXPECT_FALSE(n.equals(o, info));

    Node &info_this = info.child(0), &info_children = info.child(1);
    EXPECT_FALSE(info_this.has_child("errors"));
    EXPECT_EQ(info_children.number_of_children(), n_num_children);

    for(index_t leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        EXPECT_TRUE(info_children.child(leaf_idx).dtype().is_list());
        EXPECT_EQ(info_children.child(leaf_idx).number_of_children(), 2);

        Node &info_child = info_children.child(leaf_idx).child(0);
        EXPECT_EQ(info_child.has_child("errors"), leaf_idx % 2 == 1);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, equals_list_size_diff)
{
    const index_t n_num_children = 5;
    Node n, info;
    for(index_t leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        n.append().set(leaf_idx);
    }

    { // Full vs. Empty Node Check //
        Node o(DataType::list()), info;
        EXPECT_FALSE(n.equals(o, info));

        Node &info_this = info.child(0), &info_children = info.child(1);
        EXPECT_TRUE(info_this.has_child("errors"));
        EXPECT_EQ(info_children.number_of_children(), 0);

        Node &info_errors = info_this.fetch("errors");
        EXPECT_EQ(info_errors.number_of_children(), n_num_children);
        for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
        {
            Node &info_child = info_errors.child(child_idx);
            EXPECT_TRUE(info_child.dtype().is_string());
            EXPECT_NE(info_child.as_string().find("arg"), std::string::npos);
        }
    }

    { // Equal Node Check //
        Node o(n), info;
        EXPECT_TRUE(n.equals(o, info));

        Node &info_this = info.child(0), &info_children = info.child(1);
        EXPECT_FALSE(info_this.has_child("errors"));
        EXPECT_EQ(info_children.number_of_children(), n_num_children);

        for(index_t child_idx = 0; child_idx < n_num_children; child_idx++)
        {
            EXPECT_EQ(
                info_children.child(child_idx).child(0).fetch("valid").as_string(),
                "true");
        }
    }

    { // Half-Full Node Check //
        Node o(n), info;
        for(index_t child_idx = n_num_children - 1; child_idx >= n_num_children / 2; child_idx--)
        {
            o.remove(child_idx);
        }

        EXPECT_FALSE(n.equals(o, info));

        Node &info_this = info.child(0), &info_children = info.child(1);
        EXPECT_TRUE(info_this.has_child("errors"));
        EXPECT_EQ(info_this.fetch("errors").number_of_children(), n_num_children - n_num_children/2);

        Node &info_errors = info_this.fetch("errors");
        index_t child_idx = 0;
        for(; child_idx < n_num_children / 2; child_idx++)
        {
            EXPECT_EQ(
                info_children.child(child_idx).child(0).fetch("valid").as_string(),
                "true");
        }
        for(index_t error_idx = 0; child_idx < n_num_children; child_idx++, error_idx++)
        {
            Node &info_child = info_errors.child(error_idx);
            EXPECT_TRUE(info_child.dtype().is_string());
            EXPECT_NE(info_child.as_string().find("arg"), std::string::npos);
        }
    }
}
