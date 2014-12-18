//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

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
    
    std::cout << conduit::about() << std::endl;
    
}
