// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_node_move_and_swap.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_move_and_swap, swap_full_tree)
{
    Node n_a, n_b, n_a_info, n_b_info;

    n_a["here/is/my/data"]  = 10;
    n_b["i/also/have/data"] = 20;
    n_b["i/also/have/a/word"] = "word";

    void *data_ptr_a_orig = n_a["here/is/my/data"].data_ptr();
    void *data_ptr_b_orig = n_b["i/also/have/data"].data_ptr();
    void *str_ptr_b_orig  = n_b["i/also/have/a/word"].data_ptr();

    Schema *schema_ptr_a_orig = n_a.schema_ptr();
    Schema *schema_ptr_b_orig = n_b.schema_ptr();

    n_a.info(n_a_info);
    n_b.info(n_b_info);

    std::cout << "-Original-" << std::endl;
    std::cout << "Node A:" << std::endl;
    n_a_info.print();
    std::cout << "Node B:" << std::endl;
    n_b_info.print();

    n_a.swap(n_b);

    EXPECT_EQ(n_a["i/also/have/data"].to_int64(),20);
    EXPECT_EQ(n_a["i/also/have/a/word"].as_string(),"word");
    EXPECT_EQ(n_b["here/is/my/data"].to_int64(),10);

    // get new info
    n_a.info(n_a_info);
    n_b.info(n_b_info);

    std::cout << "-Swapped-" << std::endl;
    std::cout << "Node A:" << std::endl;
    n_a_info.print();
    std::cout << "Node B:" << std::endl;
    n_b_info.print();

    void *data_ptr_a_new = n_a["i/also/have/data"].data_ptr();
    void *str_ptr_a_new  = n_a["i/also/have/a/word"].data_ptr();
    void *data_ptr_b_new = n_b["here/is/my/data"].data_ptr();
    Schema *schema_ptr_a_new = n_a.schema_ptr();
    Schema *schema_ptr_b_new = n_b.schema_ptr();

    EXPECT_EQ(data_ptr_a_orig,data_ptr_b_new);
    EXPECT_EQ(data_ptr_b_orig,data_ptr_a_new);

    EXPECT_EQ(schema_ptr_a_orig,schema_ptr_b_new);
    EXPECT_EQ(schema_ptr_b_orig,schema_ptr_a_new);
    EXPECT_EQ(str_ptr_b_orig,str_ptr_a_new);


}

//-----------------------------------------------------------------------------
TEST(conduit_node_move_and_swap, swap_sub_tree)
{
    
    Node n_a, n_b, n_a_info, n_b_info;

    n_a["here/is/my/data"]  = 10;
    n_b["i/also/have/data"] = 20;

    void *data_ptr_a_orig = n_a["here/is/my/data"].data_ptr();
    void *data_ptr_b_orig = n_b["i/also/have/data"].data_ptr();

    Schema *schema_ptr_a_sub_orig = n_a["here/is/my/data"].schema_ptr();
    Schema *schema_ptr_b_sub_orig = n_b["i/also/have/data"].schema_ptr();

    n_a.info(n_a_info);
    n_b.info(n_b_info);

    std::cout << "-Original-" << std::endl;
    std::cout << "Node A:" << std::endl;
    n_a_info.print();
    std::cout << "Node B:" << std::endl;
    n_b_info.print();

    n_a["here/is/my/data"].swap(n_b["i/also/have/data"]);

    EXPECT_EQ(n_a["here/is/my/data"].to_int64(),20);
    EXPECT_EQ(n_b["i/also/have/data"].to_int64(),10);

    // get new info
    n_a.info(n_a_info);
    n_b.info(n_b_info);

    std::cout << "-Swapped-" << std::endl;
    
    std::cout << "Node A:" << std::endl;
    n_a_info.print();
    std::cout << "Node B:" << std::endl;
    n_b_info.print();

    void *data_ptr_a_new = n_a["here/is/my/data"].data_ptr();
    void *data_ptr_b_new = n_b["i/also/have/data"].data_ptr();
    Schema *schema_ptr_a_sub_new = n_a["here/is/my/data"].schema_ptr();
    Schema *schema_ptr_b_sub_new = n_b["i/also/have/data"].schema_ptr();

    EXPECT_EQ(data_ptr_a_orig,data_ptr_b_new);
    EXPECT_EQ(data_ptr_b_orig,data_ptr_a_new);

    EXPECT_EQ(schema_ptr_a_sub_orig,schema_ptr_b_sub_new);
    EXPECT_EQ(schema_ptr_b_sub_orig,schema_ptr_a_sub_new);

}

//-----------------------------------------------------------------------------
TEST(conduit_node_move_and_swap, move)
{
    
    Node n_a, n_b, n_a_info, n_b_info;

    n_b["data"]  = 10;

    void *data_ptr_b_orig = n_b["data"].data_ptr();
    Schema *schema_ptr_b_orig = n_b.schema_ptr();

    n_a.info(n_a_info);
    n_b.info(n_b_info);

    std::cout << "-Original-" << std::endl;
    std::cout << "Node A:" << std::endl;
    n_a_info.print();
    std::cout << "Node B:" << std::endl;
    n_b_info.print();

    n_a.move(n_b);

    EXPECT_TRUE(n_b.dtype().is_empty());

    EXPECT_EQ(n_a["data"].to_int64(),10);

    // get new info
    n_a.info(n_a_info);
    n_b.info(n_b_info);

    std::cout << "-Moved-" << std::endl;
    std::cout << "Node A:" << std::endl;
    n_a_info.print();
    std::cout << "Node B:" << std::endl;
    n_b_info.print();

    void *data_ptr_a_new = n_a["data"].data_ptr();
    Schema *schema_ptr_a_new = n_a.schema_ptr();

    EXPECT_EQ(data_ptr_b_orig,data_ptr_a_new);

    EXPECT_EQ(schema_ptr_b_orig,schema_ptr_a_new);


}
