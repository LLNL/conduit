//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: t_conduit_docs_tutorial_errors.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;


//-----------------------------------------------------------------------------
TEST(conduit_tutorial, error_handlers_funcs_start)
{
    CONDUIT_INFO("error_handlers_funcs");
}

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

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, error_handlers_funcs_end)
{
    CONDUIT_INFO("error_handlers_funcs");
}


//-----------------------------------------------------------------------------
TEST(conduit_tutorial, error_handlers)
{
     CONDUIT_INFO("error_handlers");

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
    
    // restore default handlers
    conduit::utils::set_info_handler(conduit::utils::default_info_handler);
    conduit::utils::set_warning_handler(conduit::utils::default_warning_handler);
    conduit::utils::set_error_handler(conduit::utils::default_error_handler);

    CONDUIT_INFO("error_handlers");
}






