// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_tutorial_errors.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "t_conduit_docs_tutorial_helpers.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;


//-----------------------------------------------------------------------------
TEST(conduit_tutorial, error_handlers_funcs_start)
{
    BEGIN_EXAMPLE("error_handlers_funcs");
}

// _conduit_error_handlers_funcs_start

//-----------------------------------------------------------------------------
void my_info_handler(const std::string &msg,
                     const std::string &file,
                     int line)
{
    std::cout << "[INFO] " << msg << std::endl;
}

void my_warning_handler(const std::string &msg,
                        const std::string &file,
                        int line)
{
    std::cout << "[WARNING!] " << msg << std::endl;
}

void my_error_handler(const std::string &msg,
                      const std::string &file,
                      int line)
{
    std::cout << "[ERROR!] " << msg << std::endl;
    // errors are considered fatal, aborting or unwinding the 
    // call stack with an exception are the only viable options
    throw conduit::Error(msg,file,line);
}

// _conduit_error_handlers_funcs_end

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, error_handlers_funcs_end)
{
    END_EXAMPLE("error_handlers_funcs");
}


//-----------------------------------------------------------------------------
TEST(conduit_tutorial, error_handlers)
{
    BEGIN_EXAMPLE("error_handlers_rewire");
    // rewire error handlers
    conduit::utils::set_info_handler(my_info_handler);
    conduit::utils::set_warning_handler(my_warning_handler);
    conduit::utils::set_error_handler(my_error_handler);

    // emit an example info message
    CONDUIT_INFO("An info message");

    Node n;
    n["my_value"].set_float64(42.0);
    
    // emit an example warning message
    
    // using "as" for wrong type emits a warning, returns a default value (0.0)
    float32 v = n["my_value"].as_float32();
    
    // emit an example error message

    try
    {
        // fetching a non-existant path from a const Node emits an error
        const Node &n_my_value = n["my_value"];
        n_my_value["bad"];
    }
    catch(conduit::Error e)
    {
        // pass
    }
    END_EXAMPLE("error_handlers_rewire");

    BEGIN_EXAMPLE("error_handlers_reset");
    // restore default handlers
    conduit::utils::set_info_handler(conduit::utils::default_info_handler);
    conduit::utils::set_warning_handler(conduit::utils::default_warning_handler);
    conduit::utils::set_error_handler(conduit::utils::default_error_handler);
    END_EXAMPLE("error_handlers_reset");
}






