// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_intro_cpp_example.cpp
///
//-----------------------------------------------------------------------------

#include <iostream>
#include "gtest/gtest.h"
#include <conduit.hpp>
using namespace conduit;

//-----------------------------------------------------------------------------
TEST(conduit_intro_talk_cpp_example, basic)
{
    std::vector<float64> den(4,1.0);
    
    conduit::Node n;
    n["fields/density/values"] = den;
    n["fields/density/units"] = "g/cc";
    
    Node &n_den = n["fields/density"];
    
    float64 *den_ptr = n_den["values"].value();
    std::string den_units = n_den["units"].as_string();
    
    n_den.print();
    
    std::cout << "\nDensity (" << den_units << "):\n";
    for(index_t i=0; i < 4; i++)
    {
        std::cout << den_ptr[i] << " ";
    }

    std::cout << std::endl;
}
