//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: t_conduit_docs_tutorial_parse.cpp
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
TEST(conduit_tutorial, t_conduit_docs_tutorial_yaml)
{
    BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml");

    std::string yaml_txt("mykey: 42.0");

    Node n;
    n.parse(yaml_txt,"yaml");

    std::cout << n["mykey"].as_float64() <<std::endl;

    n.print_detailed();

    END_EXAMPLE("t_conduit_docs_tutorial_yaml");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, t_conduit_docs_tutorial_json)
{
    BEGIN_EXAMPLE("t_conduit_docs_tutorial_json");

    std::string json_txt("{\"mykey\": 42.0}");

    Node n;
    n.parse(json_txt,"json");

    std::cout << n["mykey"].as_float64() <<std::endl;

    n.print_detailed();

    END_EXAMPLE("t_conduit_docs_tutorial_json");
}

//-----------------------------------------------------------------------------
TEST(conduit_tutorial, t_conduit_docs_tutorial_yaml_inline_array)
{
    BEGIN_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array");

    std::string yaml_txt("myarray: [0.0, 10.0, 20.0, 30.0]");

    Node n;
    n.parse(yaml_txt,"yaml");

    n["myarray"].print();

    n.print_detailed();

    END_EXAMPLE("t_conduit_docs_tutorial_yaml_inline_array");
}

