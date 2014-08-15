///
/// file: conduit_node_compact.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
#include "rapidjson/document.h"
using namespace conduit;

TEST(compact_1, conduit_node_compact)
{

    uint32   vals[] = {10,20,30,40,50,60,70,80,90,100};

    
    Generator g("{vals: {dtype:uint32, length:5, stride:8}}",vals);
    Node n(g);

    EXPECT_EQ(40,n.total_bytes());
    EXPECT_EQ(20,n.total_bytes_compact());

    Node nc;
    n.compact_to(nc);
    EXPECT_EQ(20,nc.total_bytes());
    EXPECT_EQ(20,nc.total_bytes_compact());
}
