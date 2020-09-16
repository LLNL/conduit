// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_node_iterator.cpp
///
//-----------------------------------------------------------------------------


#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_node_iterator, simple_1)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    std::cout << n.to_json();
    
    NodeIterator itr = n.children();
    Node itr_info;
    itr.info(itr_info);
    std::cout <<itr_info.to_json("conduit_json") << std::endl;
    
    index_t i = 0;
    while(itr.has_next())
    {
        Node &n = itr.next();
        
        if(i == 0)
        {
            EXPECT_EQ("a",itr.name());
            EXPECT_EQ(i,itr.index());
            EXPECT_EQ(a_val,n.as_uint32());
        }
        else if(i == 1)
        {
            EXPECT_EQ("b",itr.name());
            EXPECT_EQ(i,itr.index());
            EXPECT_EQ(b_val,n.as_uint32());
        }
        i++;
    }
    
    i = 0;
    while(itr.has_previous())
    {
        Node &n = itr.previous();
        
        if(i == 1)
        {
            EXPECT_EQ("a",itr.name());
            EXPECT_EQ(i,itr.index());
            EXPECT_EQ(a_val,n.as_uint32());
        }
        else if(i == 2)
        {
            EXPECT_EQ("b",itr.name());
            EXPECT_EQ(i,itr.index());
            EXPECT_EQ(b_val,n.as_uint32());
        }
        i++;
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_node_iterator, const_itr_from_const_node)
{
    uint32   c_val  = 10;
    uint32   d_val  = 20;

    Node n;
    n["a/b/c"] = c_val;
    n["a/b/d"] = d_val;

    const Node &n_b = n["a/b"];
    
    NodeConstIterator itr = n_b.children();
    
    const Node &n_c = itr.next();
    EXPECT_EQ(n_c.as_uint32(),c_val);
    
    const Node &n_d = itr.next();
    EXPECT_EQ(n_d.as_uint32(),d_val);
    Node itr_info;
    itr.info(itr_info);
    CONDUIT_INFO( itr_info.to_json() );
    
    

}

//-----------------------------------------------------------------------------
TEST(conduit_node_iterator, empty)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"]; // empty


    NodeIterator itr(n.children());
    
    while(itr.has_next())
    {
        Node &n = itr.next();
        CONDUIT_INFO(n.to_json());
    }
    
    itr = n["c"].children();
    EXPECT_FALSE(itr.has_next());
    
}


//-----------------------------------------------------------------------------
TEST(conduit_node_iterator, move_cursor)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = "myval";
    

    NodeIterator itr;
    
    itr = n.children();

    itr.to_back();
    EXPECT_FALSE(itr.has_next());
    
    // we are at the end,  check error capture
    EXPECT_THROW(itr.next(),conduit::Error);
    EXPECT_THROW(itr.peek_next(),conduit::Error);
    
    Node &last = itr.peek_previous();
    EXPECT_EQ(last.as_string(),"myval");
    
    itr.to_front();
    EXPECT_TRUE(itr.has_next());
    
    // we are at the beginning, check error capture
    EXPECT_THROW(itr.previous(),conduit::Error);
    EXPECT_THROW(itr.peek_previous(),conduit::Error);
    
    Node &first = itr.peek_next();
    EXPECT_EQ(first.as_uint32(),a_val);

    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_uint32(),a_val);
    EXPECT_EQ(itr.name(),"a");
    
    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_uint32(),b_val);
    EXPECT_EQ(itr.name(),"b");

    //  create new iterator with current state of itr
    NodeIterator itr_2(itr);
    
    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_string(),"myval");
    EXPECT_EQ(itr.name(),"c");
    
    EXPECT_FALSE(itr.has_next());
    
    // replaying the above commands on itr_2.
    
    EXPECT_TRUE(itr_2.has_next());
    EXPECT_EQ(itr_2.next().as_string(),"myval");
    EXPECT_EQ(itr_2.name(),"c");
    
    EXPECT_FALSE(itr_2.has_next());
    
    
    // const itr from itr
    itr.to_front();
    
    NodeConstIterator citr(itr);
    
    int count=0;

    while(citr.has_next())
    {
        const Node &curr = citr.next();
        index_t curr_idx = citr.index();
        EXPECT_EQ(n.child_ptr(curr_idx),&curr);
        count++;
    }

    EXPECT_EQ(count,n.number_of_children());
    
    // reset from non-const 
    citr = itr;

    count=0;

    while(citr.has_next())
    {
        citr.next();
        const Node &curr = citr.node();
        count++;
    }

    
    EXPECT_EQ(count,n.number_of_children());
    
    

}


//-----------------------------------------------------------------------------
TEST(conduit_node_iterator, const_move_cursor)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = "myval";
    

    NodeConstIterator itr;
    
    itr = n.children();

    itr.to_back();
    EXPECT_FALSE(itr.has_next());
    
    // we are at the end,  check error capture
    EXPECT_THROW(itr.next(),conduit::Error);
    EXPECT_THROW(itr.peek_next(),conduit::Error);
    
    const Node &last = itr.peek_previous();
    EXPECT_EQ(last.as_string(),"myval");
    
    itr.to_front();
    EXPECT_TRUE(itr.has_next());
    
    // we are at the beginning, check error capture
    EXPECT_THROW(itr.previous(),conduit::Error);
    EXPECT_THROW(itr.peek_previous(),conduit::Error);
    
    const Node &first = itr.peek_next();
    EXPECT_EQ(first.as_uint32(),a_val);

    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_uint32(),a_val);
    EXPECT_EQ(itr.name(),"a");
    
    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_uint32(),b_val);
    EXPECT_EQ(itr.name(),"b");

    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_string(),"myval");
    EXPECT_EQ(itr.name(),"c");
    
    EXPECT_FALSE(itr.has_next());


    EXPECT_TRUE(itr.has_previous());
    EXPECT_EQ(itr.previous().as_uint32(),b_val);
    EXPECT_EQ(itr.name(),"b");
    
    
    NodeConstIterator itr_2(itr);


    EXPECT_TRUE(itr_2.has_next());
    EXPECT_EQ(itr_2.next().as_string(),"myval");
    EXPECT_EQ(itr_2.name(),"c");
    
    EXPECT_FALSE(itr_2.has_next());
    
    // replay after assignment 
    itr_2 = itr;
    
    EXPECT_TRUE(itr_2.has_next());
    EXPECT_EQ(itr_2.next().as_string(),"myval");
    EXPECT_EQ(itr_2.name(),"c");
    
    EXPECT_FALSE(itr_2.has_next());
}


//-----------------------------------------------------------------------------
TEST(conduit_node_iterator, ref_constructor)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = "myval";

    NodeIterator itr(n);
    
    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_uint32(),a_val);
    EXPECT_EQ(itr.name(),"a");
    
    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_uint32(),b_val);
    EXPECT_EQ(itr.name(),"b");

    EXPECT_TRUE(itr.has_next());
    EXPECT_EQ(itr.next().as_string(),"myval");
    EXPECT_EQ(itr.name(),"c");
    
    EXPECT_FALSE(itr.has_next());


    

    NodeConstIterator citr(n);
    
    EXPECT_TRUE(citr.has_next());
    EXPECT_EQ(citr.next().as_uint32(),a_val);
    EXPECT_EQ(citr.name(),"a");
    
    EXPECT_TRUE(citr.has_next());
    EXPECT_EQ(citr.next().as_uint32(),b_val);
    EXPECT_EQ(citr.name(),"b");

    EXPECT_TRUE(citr.has_next());
    EXPECT_EQ(citr.next().as_string(),"myval");
    EXPECT_EQ(citr.name(),"c");
    
    EXPECT_FALSE(citr.has_next());

}



