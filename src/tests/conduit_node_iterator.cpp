///
/// file: conduit_json.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;


TEST(itr_simple_1, conduit_node_iterator)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    std::cout << n.to_json();
    
    NodeIterator itr = n.iterator();
    Node itr_info;
    itr.info(itr_info);
    std::cout <<itr_info.to_json(true) << std::endl;
    
    index_t i = 0;
    while(itr.has_next())
    {
        Node &n = itr.next();
        
        if(i == 0)
        {
            EXPECT_EQ("a",itr.path());
            EXPECT_EQ(i,itr.index());
            EXPECT_EQ(a_val,n.as_uint32());
        }
        else if(i == 1)
        {
            EXPECT_EQ("b",itr.path());
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
            EXPECT_EQ("a",itr.path());
            EXPECT_EQ(i,itr.index());
            EXPECT_EQ(a_val,n.as_uint32());
        }
        else if(i == 2)
        {
            EXPECT_EQ("b",itr.path());
            EXPECT_EQ(i,itr.index());
            EXPECT_EQ(b_val,n.as_uint32());
        }
        i++;
    }
}
