///
/// file: gtest_smoke.cpp
///

#include <iostream>
#include "gtest/gtest.h"
#define BUFFERSIZE 65536
#include "b64/encode.h"
#include "b64/decode.h"

TEST(libb64_smoke_test_case, libb64_smoke)
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
