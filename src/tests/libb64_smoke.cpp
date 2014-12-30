//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC
// Produced at the Lawrence Livermore National Laboratory. 
// 
// All rights reserved.
// 
// This source code cannot be distributed without further review from 
// Lawrence Livermore National Laboratory.
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: libb64_smoke.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "gtest/gtest.h"
#define BUFFERSIZE 65536
#include "b64/encode.h"
#include "b64/decode.h"

TEST(libb64_smoke, basic_use )
{
    std::string sin("test");
    
    std::istringstream iss(sin);
    std::ostringstream oss;

    base64::encoder e;
    e.encode(iss,oss);
    
    std::cout << oss.str() << std::endl;
    std::string sout = oss.str();
    
    
    
    iss.str(sout);
    iss.clear();
    oss.str("");

    base64::decoder d;

    d.decode(iss,oss);
    
    EXPECT_EQ(oss.str(),sin);
}

