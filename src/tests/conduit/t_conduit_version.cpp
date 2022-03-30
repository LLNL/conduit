// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_version.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include <sstream>
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST(conduit_version, basic_check)
{

    conduit::Node n;
    conduit::about(n);
    std::string v_string_constructed = "";
    // Note: All newer versions of conduit will have this, it's really
    // a check that this is defined properly
    #ifdef CONDUIT_VERSION_MAJOR
        conduit::int64 v_major = n["version_major"].value();
        conduit::int64 v_minor = n["version_minor"].value();
        conduit::int64 v_patch = n["version_patch"].value();
        
        std::ostringstream oss;
        oss << v_major << "." << v_minor << "." << v_patch;
        v_string_constructed = oss.str();
    #endif

    // this string should exist as a substring in the main version string
    // (main version string might have git info, etc on the end)
    std::string v_string = n["version"].as_string();
    std::cout << v_string_constructed << std::endl;
    std::cout << v_string << std::endl;

    EXPECT_TRUE(v_string.find(v_string_constructed) != std::string::npos);
}
