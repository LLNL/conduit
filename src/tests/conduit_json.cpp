///
/// file: conduit_json.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;


TEST(conduit_to_json_1, conduit_json)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);

    std::cout << n.to_json();
}

TEST(conduit_to_json_2, conduit_json)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    uint32 *arr = new uint32[5];
    for(index_t i=0;i<5;i++)
    {
        arr[i] = i*i;
    }

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["arr"].set(DataType::Arrays::uint32(5),arr);


    std::string pure_json = n.to_json(true);
    std::cout << n.to_json();
    std::cout << pure_json << std::endl;
    
    Generator g(pure_json,"json");
    Node n2(g);
    
    EXPECT_EQ(n["a"].as_uint32(),n2["a"].as_uint32());
    EXPECT_EQ(n["b"].as_uint32(),n2["b"].as_uint32());
    
}

