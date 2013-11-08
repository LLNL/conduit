///
/// file: value_type_tests.cpp
///

#include <iostream>
#include "gtest/gtest.h"

#include "conduit.h"

using namespace conduit;

TEST(value_type_test, value_type_tests)
{
    EXPECT_EQ(ValueType::EMPTY_T,0);
    EXPECT_EQ(ValueType::name(ValueType::EMPTY_T),"[empty]");
    EXPECT_EQ(ValueType::id("[empty]"),ValueType::EMPTY_T);
    EXPECT_TRUE( (ValueType::EMPTY_T != ValueType::NODE_T) );
}