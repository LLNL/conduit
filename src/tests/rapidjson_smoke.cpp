///
/// file: rapidjson_smoke.cpp
///

#include "rapidjson/document.h"
#include "gtest/gtest.h"

TEST(rapidjson_smoke_test_case, rapidjson_smoke_test)
{
    const char json[] = "{ \"hello\" : \"world\" }";

    rapidjson::Document d;
    d.Parse<0>(json);

    ASSERT_STREQ(d["hello"].GetString(),"world");
}
