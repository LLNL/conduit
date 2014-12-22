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
/// file: rapidjson_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "rapidjson/document.h"
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(rapidjson_smoke, basic_use)
{
    const char json[] = "{ \"hello\" : \"world\" }";

    rapidjson::Document d;
    d.Parse<0>(json);

    ASSERT_STREQ(d["hello"].GetString(),"world");
}
