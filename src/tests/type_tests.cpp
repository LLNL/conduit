///
/// file: value_type_tests.cpp
///

#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"


using namespace conduit;

TEST(value_type_test, value_type_tests)
{
    EXPECT_EQ(DataType::EMPTY_T,0);
    EXPECT_EQ(DataType::id_to_name(DataType::EMPTY_T),"[empty]");
    EXPECT_EQ(DataType::name_to_id("[empty]"),DataType::EMPTY_T);
    EXPECT_TRUE( (DataType::EMPTY_T != DataType::NODE_T) );
}
