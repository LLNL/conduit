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
// TODO(JRC): Update these tests to more extensively test the 'Node:diff' function
// and its behavior to only compare against its own contents.

/// Helper Functions ///

std::string to_string(index_t index)
{
    std::ostringstream oss;
    oss << index;
    return oss.str();
}

/// Wrapper Functions ///

bool compare_nodes_diff(const Node &lnode, const Node &rnode, Node &info)
{
    return lnode.diff(rnode, info, 0.0);
}

bool compare_nodes_equal(const Node &lnode, const Node &rnode, Node &info)
{
    return lnode.equals(rnode, info, 0.0);
}

typedef bool (*NodeCompFun)(const Node&, const Node&, Node&);

const NodeCompFun NODE_COMPARE_FUNS[2] = {compare_nodes_diff, compare_nodes_equal};

/// Testing Functions ///

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_basic)
{
    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        { // Self-Similarity Test //
            Node n, info;
            n.set("");
            EXPECT_EQ(compare_nodes(n, n, info), compare_same_result);
        }

        { // Basic Difference Test //
            Node n, o, info;
            n.set(1); o.set(2);
            EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);
        }

        { // Complex Difference Test //
            Node n, o, info;

            int data[3] = {1, 2, 3};
            n.set(data, 2);
            o.set(data, 3);

            if(compare_nodes == compare_nodes_diff)
            {
                EXPECT_EQ(compare_nodes(n, o, info), compare_same_result);
                EXPECT_EQ(compare_nodes(o, n, info), !compare_same_result);
            }
            else
            {
                EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_leaf_numeric)
{
    const DataType::TypeID leaf_tids[10] = {
        DataType::INT8_ID, DataType::INT16_ID, DataType::INT32_ID, DataType::INT64_ID,
        DataType::UINT8_ID, DataType::UINT16_ID, DataType::UINT32_ID, DataType::UINT64_ID,
        DataType::FLOAT32_ID, DataType::FLOAT64_ID
    };

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        for(index_t ti = 0; ti < 10; ti++)
        {
            const DataType::TypeID leaf_tid = leaf_tids[ti];
            DataType leaf_type(leaf_tid, 5);

            const index_t type_bytes = leaf_type.stride();
            const index_t leaf_bytes = leaf_type.bytes_compact();

            conduit_byte* n_data = new conduit_byte[leaf_bytes];
            memset(n_data, 0, leaf_bytes);
            Node n(leaf_type, (void*)n_data, true);

            conduit_byte* o_data = new conduit_byte[leaf_bytes];
            memcpy(o_data, n_data, leaf_bytes);
            Node o(leaf_type, (void*)o_data, true);

            { // Leaf Similarity Test //
                Node info;
                EXPECT_EQ(compare_nodes(n, o, info), compare_same_result);
            }

            { // Leaf Difference Test //
                Node info;
                memset(&o_data[0*type_bytes], 1, 1);
                memset(&o_data[4*type_bytes], 1, 1);
                EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);

                Node &info_diff = info.child(1);
                EXPECT_EQ(info_diff.dtype().id(), leaf_tid);
                EXPECT_EQ(info_diff.dtype().number_of_elements(), 5);
                for(index_t vi = 0; vi < 5; vi++)
                {
                    bool should_uneq = vi == 0 || vi == 4;
                    bool are_uneq = memcmp(&n_data[vi*type_bytes],
                                           &o_data[vi*type_bytes], type_bytes);
                    EXPECT_EQ(are_uneq, should_uneq);
                }
            }

            delete [] n_data;
            delete [] o_data;
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_leaf_string)
{
    const std::string compare_strs[4] = {"I", "me", "You", "tHeM"};

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        for(index_t ci = 0; ci < 4; ci++)
        {
            std::string comp_str = compare_strs[ci];
            std::string diff_str = comp_str;
            {
                index_t test_len = comp_str.length();
                diff_str[test_len-1] += 1;
            }

            { // String Similarity Test //
                Node n, o, info;
                n.set(comp_str);
                o.set(comp_str);
                EXPECT_EQ(compare_nodes(n, o, info), compare_same_result);
            }

            { // String Difference Test //
                Node n, o, info;
                n.set(comp_str);
                o.set(diff_str);
                EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_leaf_mismatch)
{
    const DataType::TypeID leaf_tids[6] = {
        DataType::INT32_ID, DataType::INT64_ID,
        DataType::UINT32_ID, DataType::UINT64_ID,
        DataType::FLOAT32_ID, DataType::FLOAT64_ID
    };

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        for(index_t ti = 0; ti < 6; ti++)
        {
            const DataType::TypeID curr_tid = leaf_tids[(ti+0)%6];
            const DataType::TypeID next_tid = leaf_tids[(ti+1)%6];

            DataType curr_type(curr_tid, 1);
            DataType next_type(next_tid, 1);

            const index_t max_bytes = std::max(
                curr_type.bytes_compact(), next_type.bytes_compact());
            conduit_byte* max_data = new conduit_byte[max_bytes];
            memset(max_data, 0, max_bytes);

            Node n(curr_type, (void*)max_data, true);
            Node o(next_type, (void*)max_data, true);
            Node info;

            EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);
            EXPECT_EQ(compare_nodes(o, n, info), !compare_same_result);

            delete [] max_data;
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_object_item_diff)
{
    const index_t n_num_children = 5;

    Node n_ref, o_ref;
    for(index_t ci = 0; ci < n_num_children; ci++)
    {
        std::string cs = to_string(ci);
        n_ref[cs].set(ci);
        o_ref[cs].set(ci+(ci%2));
    }

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        const Node n(n_ref), o(o_ref);
        Node info;
        EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);

        Node &info_this = info.child(0), &info_children = info.child(1);
        for(index_t ci = 0; ci < n_num_children; ci++)
        {
            std::string cs = to_string(ci);
            EXPECT_TRUE(info_children.has_child(cs));
            EXPECT_EQ(
                info_children.fetch(cs).child(0).fetch("valid").as_string(),
                (ci % 2 == 0) ? "true" : "false");
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_object_size_diff)
{
    const index_t n_num_children = 5;

    Node n_ref;
    for(index_t ci = 0; ci < n_num_children; ci++)
    {
        n_ref[to_string(ci)].set(ci);
    }

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        { // Full vs. Empty Node Test //
            Node n(n_ref), o(DataType::object()), info;
            EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);

            Node &info_this = info.child(0), &info_children = info.child(1);
            EXPECT_TRUE(info_this.has_child("errors"));
            EXPECT_EQ(info_children.number_of_children(), 0);

            Node &info_errors = info_this.fetch("errors");
            EXPECT_EQ(info_errors.number_of_children(), n_num_children);
            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                Node &info_child = info_errors.child(ci);
                EXPECT_TRUE(info_child.dtype().is_string());
                EXPECT_NE(info_child.as_string().find("arg"), std::string::npos);
            }
        }

        { // Equal Node Test //
            Node n(n_ref), o(n_ref), info;
            EXPECT_EQ(compare_nodes(n, o, info), compare_same_result);

            Node &info_this = info.child(0), &info_children = info.child(1);
            EXPECT_FALSE(info_this.has_child("errors"));
            EXPECT_EQ(info_children.number_of_children(), n_num_children);

            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                std::string cs = to_string(ci);
                EXPECT_TRUE(info_children.has_child(cs));
            }
        }

        { // Half-Full Node Check //
            Node n(n_ref), o(n_ref), info;
            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                if(ci % 2 == 1)
                {
                    o.remove(to_string(ci));
                }
            }

            EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);

            Node &info_this = info.child(0), &info_children = info.child(1);
            EXPECT_TRUE(info_this.has_child("errors"));
            EXPECT_EQ(info_this.fetch("errors").number_of_children(), n_num_children/2);

            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                if(ci % 2 != 1)
                {
                    EXPECT_TRUE(info_children.has_child(to_string(ci)));
                }
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_list_item_diff)
{
    const index_t n_num_children = 5;
    Node n_ref, o_ref, info;
    for(index_t leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        n_ref.append().set(leaf_idx);
        o_ref.append().set(leaf_idx+(leaf_idx%2));
    }

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        const Node n(n_ref), o(o_ref);
        Node info;
        EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);

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
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_list_size_diff)
{
    const index_t n_num_children = 5;

    Node n_ref;
    for(index_t leaf_idx = 0; leaf_idx < n_num_children; leaf_idx++)
    {
        n_ref.append().set(leaf_idx);
    }

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeCompFun compare_nodes = NODE_COMPARE_FUNS[fi];
        bool compare_same_result = static_cast<bool>(fi);

        { // Full vs. Empty Node Check //
            Node n(n_ref), o(DataType::list()), info;
            EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);

            Node &info_this = info.child(0), &info_children = info.child(1);
            EXPECT_TRUE(info_this.has_child("errors"));
            EXPECT_EQ(info_children.number_of_children(), 0);

            Node &info_errors = info_this.fetch("errors");
            EXPECT_EQ(info_errors.number_of_children(), n_num_children);
            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                Node &info_child = info_errors.child(ci);
                EXPECT_TRUE(info_child.dtype().is_string());
                EXPECT_NE(info_child.as_string().find("arg"), std::string::npos);
            }
        }

        { // Equal Node Check //
            Node n(n_ref), o(n_ref), info;
            EXPECT_EQ(compare_nodes(n, o, info), compare_same_result);

            Node &info_this = info.child(0), &info_children = info.child(1);
            EXPECT_FALSE(info_this.has_child("errors"));
            EXPECT_EQ(info_children.number_of_children(), n_num_children);

            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                EXPECT_EQ(
                    info_children.child(ci).child(0).fetch("valid").as_string(),
                    "true");
            }
        }

        { // Half-Full Node Check //
            Node n(n_ref), o(n_ref), info;
            for(index_t ci = n_num_children - 1; ci >= n_num_children / 2; ci--)
            {
                o.remove(ci);
            }

            EXPECT_EQ(compare_nodes(n, o, info), !compare_same_result);

            Node &info_this = info.child(0), &info_children = info.child(1);
            EXPECT_TRUE(info_this.has_child("errors"));
            EXPECT_EQ(info_this.fetch("errors").number_of_children(), n_num_children - n_num_children/2);

            Node &info_errors = info_this.fetch("errors");
            index_t ci = 0;
            for(; ci < n_num_children / 2; ci++)
            {
                EXPECT_EQ(
                    info_children.child(ci).child(0).fetch("valid").as_string(),
                    "true");
            }
            for(index_t ei = 0; ci < n_num_children; ci++, ei++)
            {
                Node &info_child = info_errors.child(ei);
                EXPECT_TRUE(info_child.dtype().is_string());
                EXPECT_NE(info_child.as_string().find("arg"), std::string::npos);
            }
        }
    }
}
