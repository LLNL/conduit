///
/// file: conduit_smoke.cpp
///


#include "conduit.h"

#include <iostream>
#include "gtest/gtest.h"

TEST(conduit_smoke_test, conduit_smoke)
{
    EXPECT_EQ(sizeof(conduit::uint32),4);
    EXPECT_EQ(sizeof(conduit::uint64),8);
    EXPECT_EQ(sizeof(conduit::float64),8);
}
