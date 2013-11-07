///
/// file: gtest_smoke.cpp
///

#include <iostream>
#include "gtest/gtest.h"

TEST(gtest_smoke_test_case, gtest_smoke)
{
    EXPECT_EQ(1, 1);
}