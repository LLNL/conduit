///
/// file: conduit_to_string.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

TEST(conduit_node_simple_path, conduit_node)
{
    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    char *data = new char[16];
    memcpy(&data[0],&a_val,4);
    memcpy(&data[4],&b_val,4);
    memcpy(&data[8],&c_val,8);

    Schema schema("{\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}");
    Node n(schema,data);
    std::cout << n.schema().to_json() <<std::endl; 
    
    std::cout << n.to_json(true) << std::endl;
    
    EXPECT_EQ(std::string("{ \"a\": 10, \"b\": 20, \"c\": 30}\n"),n.to_json(true));
    


    Schema schema2("{\"g\": {\"a\":\"uint32\",\"b\":\"uint32\",\"c\":\"float64\"}}");
    Node n2(schema2,data);
    std::cout << n2.schema().to_json() <<std::endl; 
    std::cout << n2.to_json(true) << std::endl;
    
    EXPECT_EQ(std::string("{ \"g\": { \"a\": 10, \"b\": 20, \"c\": 30}\n}\n"),n2.to_json(true));
}

