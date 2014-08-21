///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"
// #include "rapidjson/document.h"
using namespace conduit;





TEST(conduit_node_serialize_test_1, conduit_node)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);


    n.schema().print();
    std::vector<uint8> bytes;
    Schema s_schema;
    n.serialize(bytes);
    n.schema().compact_to(s_schema);
    
    s_schema.print();

    std::cout << *((uint32*)&bytes[0]) << std::endl;
    //Schema sch(jschema);
	Node n2(s_schema,&bytes[0]);
    EXPECT_EQ(n2["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["b"].as_uint32(),b_val);
}


/*
TEST(conduit_node_serialize_test_2, conduit_node)
{

    uint32   a_val  = 10;
    uint32   b_val  = 20;
    float64  c_val  = 30.0;

    Node n;
    n["a"] = a_val;
    n["b"] = b_val;
    n["c"] = c_val;
    n["here"]["there"] = a_val;

    EXPECT_EQ(n["a"].as_uint32(),a_val);
    EXPECT_EQ(n["b"].as_uint32(),b_val);
    EXPECT_EQ(n["c"].as_float64(),c_val);
    EXPECT_EQ(n["here"]["there"].as_uint32(),a_val);

    std::string schema = n.schema();
    std::cout << "SCHEMA:\n" << schema;
    std::vector<uint8> bytes;
    n.serialize(bytes);


    Node n2(&bytes[0],schema);
    std::cout << "NEWSCHEMA:\n" << n2.schema();
    Node &t = n2["a"];
    EXPECT_EQ(n2["a"].as_uint32(),a_val);
    EXPECT_EQ(n2["b"].as_uint32(),b_val);
    EXPECT_EQ(n2["c"].as_float64(),c_val);
    EXPECT_EQ(n2["here"]["there"].as_uint32(),a_val);

}
*/
