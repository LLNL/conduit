// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: rapidjson_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_json.hpp"
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(json_smoke, basic_use)
{
    const char json[] = "{ \"hello\" : \"world\" }";

    conduit_json::Document d;
    d.Parse<0>(json);

    ASSERT_STREQ(d["hello"].GetString(),"world");
}
