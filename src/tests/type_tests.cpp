///
/// file: value_type_tests.cpp
///

#include <iostream>
#include "gtest/gtest.h"

#include "conduit.h"

using namespace conduit;

TEST(value_type_test, value_type_tests)
{
    EXPECT_EQ(BaseType::EMPTY_T,0);
    EXPECT_EQ(BaseType::type_id_to_name(BaseType::EMPTY_T),"[empty]");
    EXPECT_EQ(BaseType::type_name_to_id("[empty]"),BaseType::EMPTY_T);
    EXPECT_TRUE( (BaseType::EMPTY_T != BaseType::NODE_T) );
}