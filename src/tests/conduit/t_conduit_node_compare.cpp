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
#include <string.h>
#include "gtest/gtest.h"

using namespace conduit;

// TODO(JRC): Update these tests to more extensively test the 'Node:diff_compatible'
// function and its behavior to only compare against its own contents.

/// Helper Functions ///

std::string to_string(index_t index)
{
    std::ostringstream oss;
    oss << index;
    return oss.str();
}

/// Wrapper Functions ///

bool diff_nodes(const Node &lnode, const Node &rnode, Node &info)
{
    return lnode.diff(rnode, info, 0.0);
}

bool compatible_diff_nodes(const Node &lnode, const Node &rnode, Node &info)
{
    return lnode.diff_compatible(rnode, info, 0.0);
}

typedef bool (*NodeDiffFun)(const Node&, const Node&, Node&);

const NodeDiffFun NODE_DIFF_FUNS[2] = {diff_nodes, compatible_diff_nodes};

/// Testing Functions ///

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_basic)
{
    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        { // Self-Similarity Test //
            Node n, info;
            n.set("");
            EXPECT_FALSE(diff_nodes(n, n, info));
        }

        { // Basic Difference Test //
            Node n, o, info;
            n.set(1); o.set(2);
            EXPECT_TRUE(diff_nodes(n, o, info));
        }

        { // Complex Difference Test //
            Node n, o, info;

            int data[3] = {1, 2, 3};
            n.set(data, 2);
            o.set(data, 3);

            if(diff_nodes == compatible_diff_nodes)
            {
                EXPECT_FALSE(diff_nodes(n, o, info));
                EXPECT_TRUE(diff_nodes(o, n, info));
            }
            else
            {
                EXPECT_TRUE(diff_nodes(n, o, info));
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
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        for(index_t ti = 0; ti < 10; ti++)
        {
            const DataType::TypeID leaf_tid = leaf_tids[ti];
            DataType leaf_type(leaf_tid, 5, 5 * DataType::default_bytes(leaf_tid),
                DataType::default_bytes(leaf_tid), DataType::default_bytes(leaf_tid),
                Endianness::DEFAULT_ID);

            const index_t type_bytes = leaf_type.stride();
            const index_t leaf_bytes = leaf_type.spanned_bytes();

            int8* n_data = new int8[(size_t)leaf_bytes];
            memset(n_data, 0, (size_t)leaf_bytes);
            Node n(leaf_type, (void*)n_data, true);

            int8* o_data = new int8[(size_t)leaf_bytes];
            memcpy(o_data, n_data, (size_t)leaf_bytes);
            Node o(leaf_type, (void*)o_data, true);

            { // Leaf Similarity Test //
                Node info;
                EXPECT_FALSE(diff_nodes(n, o, info));
            }

            { // Leaf Difference Test //
                Node info;
                memset(o.element_ptr(0), 1, 1);
                memset(o.element_ptr(4), 1, 1);
                EXPECT_TRUE(diff_nodes(n, o, info));

                Node &info_diff = info["value"];
                EXPECT_EQ(info_diff.dtype().id(), leaf_tid);
                EXPECT_EQ(info_diff.dtype().number_of_elements(), 5);
                for(index_t vi = 0; vi < 5; vi++)
                {
                    bool should_uneq = vi == 0 || vi == 4;
                    bool are_uneq = (0 != memcmp(n.element_ptr(vi),
                                                 o.element_ptr(vi),
                                                 (size_t)type_bytes));
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
    char compare_buffs[5][10];

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        for(index_t ci = 0; ci < 4; ci++)
        {
            index_t leaf_tid = DataType::CHAR8_STR_ID;
            std::string leaf_str = compare_strs[ci];
            // NOTE(JRC): This test applies a buffer offset to the data being
            // tested to push the data to the end of the buffer. For an
            // example test phrase "me", this buffer looks like the following:
            //         leaf_buff
            //   [ _ _ _ _ _ _ _ m e / ]
            //     0 1 2 3 4 5 6 7 8 9
            DataType leaf_type(leaf_tid, leaf_str.length() + 1,
                (10 - leaf_str.length() - 1) * DataType::default_bytes(leaf_tid),
                DataType::default_bytes(leaf_tid), DataType::default_bytes(leaf_tid),
                Endianness::DEFAULT_ID);

            char* leaf_buff = (char*)&compare_buffs[ci];
            char* leaf_cstr = (char*)&compare_buffs[ci+1] - leaf_str.length() - 1;
            size_t leaf_buff_size = (size_t)leaf_type.spanned_bytes();
            memset(leaf_buff, 0, leaf_buff_size);
            snprintf(leaf_cstr, leaf_buff_size, "%s", leaf_str.c_str());

            Node n(leaf_type, (void*)leaf_buff, true);

            { // String Similarity Test //
                Node  o(leaf_type, (void*)leaf_buff, true), info;
                EXPECT_FALSE(diff_nodes(n, o, info));
            }

            { // String Difference Test //
                char* diff_buff = (char*)&compare_buffs[4];
                memcpy(diff_buff, leaf_buff, (size_t)leaf_type.spanned_bytes());
                diff_buff[8] += 1;

                Node  o(leaf_type, (void*)diff_buff, true), info;
                EXPECT_TRUE(diff_nodes(n, o, info));
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
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        for(index_t ti = 0; ti < 6; ti++)
        {
            const DataType::TypeID curr_tid = leaf_tids[(ti+0)%6];
            const DataType::TypeID next_tid = leaf_tids[(ti+1)%6];

            DataType curr_type(curr_tid, 1);
            DataType next_type(next_tid, 1);

            const index_t max_bytes = std::max(
                curr_type.bytes_compact(), next_type.bytes_compact());
            int8* max_data = new int8[(size_t)max_bytes];
            memset(max_data, 0, (size_t)max_bytes);

            Node n(curr_type, (void*)max_data, true);
            Node o(next_type, (void*)max_data, true);
            Node info;

            EXPECT_TRUE(diff_nodes(n, o, info));
            EXPECT_TRUE(diff_nodes(o, n, info));

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
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        const Node n(n_ref), o(o_ref);
        Node info;
        EXPECT_TRUE(diff_nodes(n, o, info));

        Node &info_children = info["children/diff"];
        for(index_t ci = 0; ci < n_num_children; ci++)
        {
            std::string cs = to_string(ci);
            EXPECT_TRUE(info_children.has_child(cs));
            EXPECT_EQ(
                info_children.fetch(cs).fetch("valid").as_string(),
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
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        { // Full vs. Empty Node Test //
            Node n(n_ref), o(DataType::object()), info;
            EXPECT_TRUE(diff_nodes(n, o, info));

            EXPECT_EQ(info["valid"].as_string(), "false");
            EXPECT_TRUE(info["children"].has_child("extra"));
            EXPECT_FALSE(info["children"].has_child("missing"));
            EXPECT_FALSE(info["children"].has_child("diff"));

            Node &info_extra = info["children/extra"];
            EXPECT_EQ(info_extra.number_of_children(), n_num_children);
        }

        { // Equal Node Test //
            Node n(n_ref), o(n_ref), info;
            EXPECT_FALSE(diff_nodes(n, o, info));

            EXPECT_EQ(info["valid"].as_string(), "true");
            EXPECT_TRUE(info["children"].has_child("diff"));
            EXPECT_FALSE(info["children"].has_child("extra"));
            EXPECT_FALSE(info["children"].has_child("missing"));

            Node &info_diff = info["children/diff"];
            EXPECT_EQ(info_diff.number_of_children(), n_num_children);

            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                std::string cs = to_string(ci);
                EXPECT_TRUE(info_diff.has_child(cs));
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

            EXPECT_TRUE(diff_nodes(n, o, info));

            EXPECT_EQ(info["valid"].as_string(), "false");
            EXPECT_TRUE(info["children"].has_child("diff"));
            EXPECT_TRUE(info["children"].has_child("extra"));
            EXPECT_FALSE(info["children"].has_child("missing"));

            Node &info_extra = info["children/extra"];
            EXPECT_EQ(info_extra.number_of_children(), n_num_children/2);

            Node &info_diff = info["children/diff"];
            for(index_t ci = 0; ci < n_num_children; ci++)
            {
                if(ci % 2 != 1)
                {
                    EXPECT_TRUE(info_diff.has_child(to_string(ci)));
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
    for(index_t ci = 0; ci < n_num_children; ci++)
    {
        n_ref.append().set(ci);
        o_ref.append().set(ci+(ci%2));
    }

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        const Node n(n_ref), o(o_ref);
        Node info;
        EXPECT_TRUE(diff_nodes(n, o, info));

        Node &info_children = info["children/diff"];
        for(index_t ci = 0; ci < n_num_children; ci++)
        {
            EXPECT_EQ(
                info_children.child(ci).fetch("valid").as_string(),
                (ci % 2 == 0) ? "true" : "false");
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_compare, compare_list_size_diff)
{
    const index_t n_num_children = 5;

    Node n_ref;
    for(index_t ci = 0; ci < n_num_children; ci++)
    {
        n_ref.append().set(ci);
    }

    for(index_t fi = 0; fi < 2; fi++)
    {
        NodeDiffFun diff_nodes = NODE_DIFF_FUNS[fi];

        { // Full vs. Empty Node Check //
            Node n(n_ref), o(DataType::list()), info;
            EXPECT_TRUE(diff_nodes(n, o, info));

            EXPECT_EQ(info["valid"].as_string(), "false");
            EXPECT_TRUE(info["children"].has_child("extra"));
            EXPECT_FALSE(info["children"].has_child("missing"));
            EXPECT_FALSE(info["children"].has_child("diff"));

            Node &info_extra = info["children/extra"];
            EXPECT_EQ(info_extra.number_of_children(), n_num_children);
        }

        { // Equal Node Check //
            Node n(n_ref), o(n_ref), info;
            EXPECT_FALSE(diff_nodes(n, o, info));

            EXPECT_EQ(info["valid"].as_string(), "true");
            EXPECT_TRUE(info["children"].has_child("diff"));
            EXPECT_FALSE(info["children"].has_child("extra"));
            EXPECT_FALSE(info["children"].has_child("missing"));

            Node &info_diff = info["children/diff"];
            EXPECT_EQ(info_diff.number_of_children(), n_num_children);
        }

        { // Half-Full Node Check //
            Node n(n_ref), o(n_ref), info;
            for(index_t ci = n_num_children - 1; ci >= n_num_children / 2; ci--)
            {
                o.remove(ci);
            }

            EXPECT_TRUE(diff_nodes(n, o, info));

            EXPECT_EQ(info["valid"].as_string(), "false");
            EXPECT_TRUE(info["children"].has_child("diff"));
            EXPECT_TRUE(info["children"].has_child("extra"));
            EXPECT_FALSE(info["children"].has_child("missing"));

            Node &info_extra = info["children/extra"];
            EXPECT_EQ(info_extra.number_of_children(), n_num_children/2+1);

            Node &info_diff = info["children/diff"];
            EXPECT_EQ(info_diff.number_of_children(), n_num_children/2);
        }
    }
}
