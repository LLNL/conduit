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
/// file: conduit_node_cpp_iterator.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------

TEST(conduit_node_cpp_iterator, non_const_types) {
    ::testing::StaticAssertTypeEq<Node, NodeChildIterator::value_type>();
    ::testing::StaticAssertTypeEq<Node &, NodeChildIterator::reference>();
    ::testing::StaticAssertTypeEq<Node *, NodeChildIterator::pointer>();
    ::testing::StaticAssertTypeEq<index_t, NodeChildIterator::difference_type>();
    ::testing::StaticAssertTypeEq<std::random_access_iterator_tag,
            NodeChildIterator::iterator_category>();

    NodeChildIterator iter;
    ::testing::StaticAssertTypeEq<Node &, decltype(*iter)>();
    ::testing::StaticAssertTypeEq<Node *, decltype(iter.operator->())>();
    ::testing::StaticAssertTypeEq<NodeChildIterator &, decltype(iter += 5)>();
    ::testing::StaticAssertTypeEq<NodeChildIterator &, decltype(iter -= 5)>();
    ::testing::StaticAssertTypeEq<NodeChildIterator, decltype(iter + 5)>();
    ::testing::StaticAssertTypeEq<NodeChildIterator, decltype(5 + iter)>();
    ::testing::StaticAssertTypeEq<NodeChildIterator, decltype(iter - 5)>();
    NodeChildIterator iter_2;
    ::testing::StaticAssertTypeEq<index_t, decltype(iter - iter_2)>();
    ::testing::StaticAssertTypeEq<Node &, decltype(iter[3])>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter < iter_2)>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter <= iter_2)>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter > iter_2)>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter >= iter_2)>();
}

TEST(conduit_node_cpp_iterator, const_types) {
    ::testing::StaticAssertTypeEq<Node, NodeConstChildIterator::value_type>();
    ::testing::StaticAssertTypeEq<Node const &,
            NodeConstChildIterator::reference>();
    ::testing::StaticAssertTypeEq<Node const *,
            NodeConstChildIterator::pointer>();
    ::testing::StaticAssertTypeEq<index_t, NodeChildIterator::difference_type>();
    ::testing::StaticAssertTypeEq<std::random_access_iterator_tag,
            NodeConstChildIterator::iterator_category>();

    NodeConstChildIterator iter;
    ::testing::StaticAssertTypeEq<Node const &, decltype(*iter)>();
    ::testing::StaticAssertTypeEq<Node const *, decltype(iter.operator->())>();
    ::testing::StaticAssertTypeEq<NodeConstChildIterator &, decltype(iter += 5)>();
    ::testing::StaticAssertTypeEq<NodeConstChildIterator &, decltype(iter -= 5)>();
    ::testing::StaticAssertTypeEq<NodeConstChildIterator, decltype(iter + 5)>();
    ::testing::StaticAssertTypeEq<NodeConstChildIterator, decltype(5 + iter)>();
    ::testing::StaticAssertTypeEq<NodeConstChildIterator, decltype(iter - 5)>();
    NodeConstChildIterator iter_2;
    ::testing::StaticAssertTypeEq<index_t, decltype(iter - iter_2)>();
    ::testing::StaticAssertTypeEq<Node const &, decltype(iter[3])>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter < iter_2)>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter <= iter_2)>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter > iter_2)>();
    ::testing::StaticAssertTypeEq<bool, decltype(iter >= iter_2)>();
}

TEST(conduit_node_cpp_iterator, conversion_from_non_const_to_const) {
    Node node;
    NodeChildIterator iter{&node, 3};
    NodeConstChildIterator const_iter = iter;
    NodeConstChildIterator expected_iter{&node, 3};
    EXPECT_EQ(const_iter, expected_iter);
}

TEST(conduit_node_cpp_iterator, cbegin_cend) {
    Node node;
    node.add_child("c1");
    NodeConstChildIterator begin = static_cast<Node const &>(node).children().begin();
    NodeConstChildIterator cbegin = node.children().cbegin();
    EXPECT_EQ(begin, cbegin);
    NodeConstChildIterator end = static_cast<Node const &>(node).children().end();
    NodeConstChildIterator cend = node.children().cend();
    EXPECT_EQ(end, cend);
}

template<typename iterator>
class conduit_node_cpp_iteration : public ::testing::Test {
protected:
    void SetUp() override {
        m_modifiable_node.add_child("c0").set("value 0");
        m_modifiable_node.add_child("c1").set("value 1");
        m_modifiable_node.add_child("c2").set("value 2");
        m_modifiable_node.add_child("c3").set("value 3");
        m_base_node = &m_modifiable_node;
    }

    typename iterator::pointer m_base_node;

    typename iterator::reference last_child() {
        return m_base_node->child(m_base_node->number_of_children() - 1);
    }

private:
    Node m_modifiable_node;
};

using IteratorTypes = ::testing::Types<NodeChildIterator, NodeConstChildIterator>;
TYPED_TEST_SUITE(conduit_node_cpp_iteration, IteratorTypes);

TYPED_TEST(conduit_node_cpp_iteration, operator_indirect) {
    TypeParam iter = this->m_base_node->children().begin();
    EXPECT_EQ((*iter).as_string(), this->m_base_node->child(0).as_string());
}

TYPED_TEST(conduit_node_cpp_iteration, operator_arrow) {
    TypeParam iter = this->m_base_node->children().begin();
    Node n;
    EXPECT_EQ(iter->as_string(), this->m_base_node->child(0).as_string());
}

TYPED_TEST(conduit_node_cpp_iteration, equality) {
    TypeParam iter{this->m_base_node};
    EXPECT_EQ(iter, TypeParam{this->m_base_node})
                        << "Different iterators to same node";
    TypeParam different_child = TypeParam{this->m_base_node, 1};
    EXPECT_NE(iter, different_child) << "Iterators to different children";

    Node different_parent;
    typename TypeParam::pointer different_parent_ptr = &different_parent;
    EXPECT_NE(iter, TypeParam{different_parent_ptr})
                        << "Same position in different nodes";
    TypeParam different_child_of_different_parent{different_parent_ptr, 1};
    EXPECT_NE(iter, different_child_of_different_parent)
                        << "Different positions in different nodes";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_pre_increment) {
    TypeParam iter = this->m_base_node->children().begin();
    ::testing::StaticAssertTypeEq<TypeParam &, decltype(++iter)>();
    TypeParam &res = ++iter;
    EXPECT_EQ(&res, &iter) << "Did not get same object";
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(1)) << "Did not advance";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_post_increment) {
    TypeParam iter = this->m_base_node->children().begin();
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(0));
    ::testing::StaticAssertTypeEq<TypeParam, decltype(iter++)>();
    TypeParam res = iter++;
    EXPECT_NE(&res, &iter) << "Got same object";
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(1)) << "Did not advance";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_pre_decrement) {
    TypeParam iter = this->m_base_node->children().end();
    ::testing::StaticAssertTypeEq<TypeParam &, decltype(--iter)>();
    TypeParam &res = --iter;
    EXPECT_EQ(&res, &iter) << "Did not get same object";
    EXPECT_EQ(&*iter, &this->last_child()) << "Did not go back";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_post_decrement) {
    TypeParam iter = this->m_base_node->children().end();
    ::testing::StaticAssertTypeEq<TypeParam, decltype(iter--)>();
    TypeParam res = iter--;
    EXPECT_NE(&res, &iter) << "Got same object";
    EXPECT_EQ(&*iter, &this->last_child()) << "Did not go back";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_plus_equals) {
    TypeParam iter = this->m_base_node->children().begin();
    TypeParam &new_iter = iter += 3;
    EXPECT_EQ(&new_iter, &iter) << "Did not get same object";
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(3))
                        << "Did not advance to right place";
    TypeParam &new_iter_2 = iter += -2;
    EXPECT_EQ(&new_iter_2, &iter) << "Did not get same object";
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(1))
                        << "Did not go back to right place";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_minus_equals) {
    TypeParam iter = this->m_base_node->children().end();
    TypeParam &new_iter = iter -= 3;
    EXPECT_EQ(&new_iter, &iter) << "Did not get same object";
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(1))
                        << "Did not go back to right place";
    TypeParam &new_iter_2 = iter -= -2;
    EXPECT_EQ(&new_iter_2, &iter) << "Did not get same object";
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(3))
                        << "Did not advance to right place";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_plus) {
    TypeParam iter = this->m_base_node->children().begin();
    TypeParam new_iter = iter + 3;
    EXPECT_EQ(&*iter, this->m_base_node->child_ptr(0))
                        << "Moved unexpectedly";
    EXPECT_EQ(&*new_iter, this->m_base_node->child_ptr(3))
                        << "Did not advance to right place";
    new_iter = 3 + iter;
    EXPECT_EQ(&*new_iter, this->m_base_node->child_ptr(3))
                        << "Did not advance to right place";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_minus_offset) {
    TypeParam iter = this->m_base_node->children().end();
    TypeParam new_iter = iter - 3;
    EXPECT_EQ(&*(--iter), this->m_base_node->child_ptr(3))
                        << "Moved unexpectedly";
    EXPECT_EQ(&*new_iter, this->m_base_node->child_ptr(1))
                        << "Did not go back to right place";
}

TYPED_TEST(conduit_node_cpp_iteration, operator_minus_iter) {
    TypeParam a = this->m_base_node->children().end();
    TypeParam b{a};
    a -= 3;
    EXPECT_EQ(3, b - a);
    EXPECT_EQ(b, a + (b - a)); // post condition from cppreference.com
}

TYPED_TEST(conduit_node_cpp_iteration, operator_subscript) {
    TypeParam iter{this->m_base_node, 1};
    EXPECT_EQ(&iter[0], this->m_base_node->child_ptr(1));
    EXPECT_EQ(&iter[2], this->m_base_node->child_ptr(3));
    EXPECT_EQ(&iter[-1], this->m_base_node->child_ptr(0));
}

TYPED_TEST(conduit_node_cpp_iteration, comparison_operators) {
    TypeParam iter_1{this->m_base_node, 1};
    TypeParam iter_3{this->m_base_node, 3};
    EXPECT_LT(iter_1, iter_3);
    EXPECT_LE(iter_1, iter_1);
    EXPECT_LE(iter_1, iter_3);
    EXPECT_GT(iter_3, iter_1);
    EXPECT_GE(iter_3, iter_1);
    EXPECT_GE(iter_3, iter_3);
}

TYPED_TEST(conduit_node_cpp_iteration, enhanced_for_loop) {
    index_t current_index = 0;
    for (typename TypeParam::reference cur_node : this->m_base_node->children()) {
        EXPECT_EQ(&cur_node, this->m_base_node->child_ptr(current_index))
                            << "Got wrong child at " << current_index;
        ++current_index;
    }
    EXPECT_EQ(current_index, this->m_base_node->number_of_children())
                        << "Did not iterate over all children";
}

TYPED_TEST(conduit_node_cpp_iteration, name) {
    auto nodeIter = this->m_base_node->children();
    TypeParam childIter = nodeIter.begin();
    while(nodeIter.has_next()) {
        Node const &node = nodeIter.next();
        EXPECT_EQ(node.name(), nodeIter.name());
        EXPECT_EQ(node.name(), childIter.name());
        ++childIter;
    }
}
