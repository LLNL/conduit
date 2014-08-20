///
/// file: conduit_json.cpp
///


#include "conduit.h"

#include <iostream>
#include <sstream>

#include "gtest/gtest.h"
using namespace conduit;


TEST(conduit_bytestr_basic, conduit_bytestr)
{
    const char *c_ta = "test string for a";
    const char *c_tb = "test string for b";

    Node n;
    n["a"] = c_ta;
    n["b"] = c_tb;
    
    std::ostringstream oss;
    oss << n["a"].as_string() << " and " << n["b"].as_string();
    
    std::string cpp_tc = oss.str();
    
    n["c"] = cpp_tc;

    EXPECT_EQ(strcmp(n["a"].as_bytestr(),c_ta),0);
    EXPECT_EQ(strcmp(n["b"].as_bytestr(),c_tb),0);
    EXPECT_EQ(n["c"].as_string(),cpp_tc);

    n.print_detailed();
}
