// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_libb64_smoke.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "gtest/gtest.h"
#define BUFFERSIZE 65536
#include "b64/encode.h"
#include "b64/decode.h"


std::string
encode_decode(const std::string &val)
{
    std::string sin(val);
    
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
    
    return oss.str();
}


TEST(libb64_smoke, basic_use )
{
    std::string t = "test";
    EXPECT_EQ(encode_decode(t),t);

    // test a shorter string ...
    t = "t";
    EXPECT_EQ(encode_decode(t),t);

    // test an empty string
    t = "";
    EXPECT_EQ(encode_decode(t),t);

    t = "test a longer string";
    EXPECT_EQ(encode_decode(t),t);

    t = "test an even longer string with more vowels.";
    EXPECT_EQ(encode_decode(t),t);

}

