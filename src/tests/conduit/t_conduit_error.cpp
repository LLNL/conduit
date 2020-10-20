// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_error.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_error, error_print)
{
    Node n;
    n["some/crazy/path"] = 3.1415;
    n["some/other/crazy/path"] = 42;
    n["some/details"] = "imporant string";
    Error e(n.to_json(),"myfile.cpp" ,100);

    std::cout << e.message() << std::endl;
}

