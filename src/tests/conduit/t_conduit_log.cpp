// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_log.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_log.hpp"

#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

/// Testing Constants ///

typedef void (*LogFun)(Node&, const std::string&, const std::string&);
typedef void (*FilerFun)(Node&);

/// Testing Functions ///

//-----------------------------------------------------------------------------
TEST(conduit_log, log_functions)
{
    LogFun log_functions[] = {log::info, log::optional, log::error};
    const std::string log_headers[] = {"info", "optional", "errors"};
    const index_t fun_count = sizeof(log_functions) / sizeof(log_functions[0]);

    const std::string test_prototypes[] = {"p1", "p2"};
    const std::string test_messages[] = {"m1", "m2"};
    const index_t test_count = sizeof(test_prototypes) / sizeof(test_prototypes[0]);

    for(index_t fi = 0; fi < fun_count; fi++)
    {
        LogFun log_fun = log_functions[fi];
        const std::string &log_header = log_headers[fi];

        Node info;
        for(index_t ti = 0; ti < test_count; ti++)
        {
            log_fun(info, test_prototypes[ti], test_messages[ti]);

            ASSERT_TRUE(info.dtype().is_object());
            ASSERT_EQ(info.number_of_children(), 1);

            ASSERT_TRUE(info.has_child(log_header));
            ASSERT_TRUE(info[log_header].dtype().is_list());
            ASSERT_EQ(info[log_header].number_of_children(), ti+1);

            ASSERT_TRUE(info[log_header].child(ti).dtype().is_string());
        }

        ASSERT_NE(info[log_header].child(0).as_string(), info[log_header].child(1).as_string());
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_log, validation_functions)
{
    for(index_t ti = 0; ti < 2; ti++)
    {
        const bool test_valid = ti == 0;
        const std::string test_validity_str = test_valid ? "true" : "false";

        Node info;
        log::validation(info, test_valid);

        ASSERT_TRUE(info.dtype().is_object());
        ASSERT_EQ(info.number_of_children(), 1);

        ASSERT_TRUE(info.has_child("valid"));
        ASSERT_TRUE(info["valid"].dtype().is_string());
        ASSERT_EQ(info["valid"].as_string(), test_validity_str);

        log::validation(info, true);
        ASSERT_EQ(info.number_of_children(), 1);
        ASSERT_EQ(info["valid"].as_string(), test_validity_str);

        log::validation(info, false);
        ASSERT_EQ(info.number_of_children(), 1);
        ASSERT_EQ(info["valid"].as_string(), "false");
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_log, remove_valid_function)
{
    { // Test: One-Level Filtered/Unfiltered //
        for(index_t ti = 0; ti < 2; ti++)
        {
            const bool test_valid = ti == 0;
            const std::string test_validity_str = test_valid ? "true" : "false";

            std::ostringstream oss;
            oss << "{\"valid\": \"" << test_validity_str << "\"}";
            std::string trivial_schema = oss.str();

            Generator gen_info(trivial_schema, "json");
            Node info(gen_info, true);
            log::remove_valid(info);

            ASSERT_EQ(info.dtype().is_empty(), test_valid);
        }
    }

    { // Test: Multi-Level, Top-Level Filtered //
        std::string basic_schema = "{\"valid\": \"true\", \"a\": {\"valid\": \"true\"}, \"b\": {\"valid\": \"true\"}, \"c\": 4}";

        Generator gen_info(basic_schema, "json");
        Node info(gen_info, true);
        log::remove_valid(info);

        ASSERT_TRUE(info.dtype().is_empty());
    }

    { // Test: Multi-Level, Top-Level Unfiltered //
        std::string nontrivial_schema = "{\"valid\": \"false\", \"a\": {\"valid\": \"false\"}, \"b\": {\"valid\": \"true\"}, \"c\": 4}";

        Generator gen_info(nontrivial_schema, "json");
        Node info(gen_info, true);
        log::remove_valid(info);

        ASSERT_TRUE(info.dtype().is_object());
        ASSERT_TRUE(info.has_child("valid"));
        ASSERT_EQ(info["valid"].as_string(), "false");

        ASSERT_TRUE(info.has_child("a"));
        ASSERT_TRUE(info["a"].has_child("valid"));
        ASSERT_EQ(info["a/valid"].as_string(), "false");

        ASSERT_FALSE(info.has_child("b"));

        ASSERT_TRUE(info.has_child("c"));
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_log, remove_invalid_function)
{
    { // Test: One-Level Filtered/Unfiltered //
        for(index_t ti = 0; ti < 2; ti++)
        {
            const bool test_valid = ti == 0;
            const std::string test_validity_str = test_valid ? "true" : "false";

            std::ostringstream oss;
            oss << "{\"valid\": \"" << test_validity_str << "\"}";
            std::string trivial_schema = oss.str();

            Generator gen_info(trivial_schema, "json");
            Node info(gen_info, true);
            log::remove_invalid(info);

            ASSERT_EQ(info.dtype().is_empty(), !test_valid);
        }
    }

    { // Test: Multi-Level, Top-Level Filtered //
        std::string basic_schema = "{\"valid\": \"false\", \"a\": {\"valid\": \"false\"}, \"b\": {\"valid\": \"true\"}, \"c\": 4}";

        Generator gen_info(basic_schema, "json");
        Node info(gen_info, true);
        log::remove_invalid(info);

        ASSERT_TRUE(info.dtype().is_empty());
    }

    { // Test: Multi-Level, Top-Level Unfiltered //
        std::string nontrivial_schema = "{\"valid\": \"true\", \"a\": {\"valid\": \"true\"}, \"b\": {\"valid\": \"false\"}, \"c\": 4}";

        Generator gen_info(nontrivial_schema, "json");
        Node info(gen_info, true);
        log::remove_invalid(info);

        ASSERT_TRUE(info.dtype().is_object());
        ASSERT_TRUE(info.has_child("valid"));
        ASSERT_EQ(info["valid"].as_string(), "true");

        ASSERT_TRUE(info.has_child("a"));
        ASSERT_TRUE(info["a"].has_child("valid"));
        ASSERT_EQ(info["a/valid"].as_string(), "true");

        ASSERT_FALSE(info.has_child("b"));

        ASSERT_TRUE(info.has_child("c"));
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_log, remove_optional_function)
{
    { // Test: One-Level Filtering //
        Node info;
        log::info(info, "", "");
        log::optional(info, "", ""),
        log::error(info, "", "");
        log::validation(info, true);
        log::remove_optional(info);

        ASSERT_TRUE(info.dtype().is_object());
        ASSERT_EQ(info.number_of_children(), 3);
        ASSERT_TRUE(info.has_child("info"));
        ASSERT_TRUE(info.has_child("errors"));
        ASSERT_TRUE(info.has_child("valid"));
    }

    { // Test: Multi-Level Filtering //
        Node info;
        log::optional(info, "", ""),
        log::info(info["a"], "", "");
        log::optional(info["a"], "", ""),
        log::info(info["b"], "", "");
        log::remove_optional(info);

        ASSERT_TRUE(info.dtype().is_object());
        ASSERT_EQ(info.number_of_children(), 2);
        ASSERT_TRUE(info.has_child("a"));
        ASSERT_TRUE(info.has_child("b"));

        ASSERT_EQ(info["a"].number_of_children(), 1);
        ASSERT_TRUE(info["a"].has_child("info"));
        ASSERT_EQ(info["b"].number_of_children(), 1);
        ASSERT_TRUE(info["b"].has_child("info"));
    }
}
