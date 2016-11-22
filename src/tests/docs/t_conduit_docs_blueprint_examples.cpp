//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://llnl.github.io/conduit/.
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
/// file: t_conduit_docs_blueprint_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"

#include <iostream>
#include "gtest/gtest.h"
using namespace conduit;

//-----------------------------------------------------------------------------
// 65-79
TEST(conduit_docs, blueprint_example_1)
{
    CONDUIT_INFO("blueprint_example_1");

    // setup our candidate and info nodes
    Node n, info;

    //create an example mesh
    conduit::blueprint::mesh::examples::braid("tets",
                                               5,5,5,
                                               n);
    // check if n conforms
    if(conduit::blueprint::verify("mesh",n,info))
        std::cout << "mesh verify succeeded." << std::endl;
    else
        std::cout << "mesh verify failed!" << std::endl;
   
    // show some of the verify details
    info["coordsets"].print();
    
    CONDUIT_INFO("blueprint_example_1");
}

//-----------------------------------------------------------------------------
// 90-110
TEST(conduit_docs, blueprint_example_2)
{
    CONDUIT_INFO("blueprint_example_2");

    // setup our candidate and info nodes
    Node n, verify_info, mem_info;

    // create an example mcarray
    conduit::blueprint::mcarray::examples::xyz("separate",5,n);
    
    std::cout << "example 'separate' mcarray " << std::endl;
    n.print();
    n.info(mem_info);
    mem_info.print();
    
    // check if n conforms
    if(conduit::blueprint::verify("mcarray",n,verify_info))
    {
        // check if our mcarray has a specific memory layout 
        if(!conduit::blueprint::mcarray::is_interleaved(n))
        {
            // copy data from n into the desired memory layout
            Node xform;
            conduit::blueprint::mcarray::to_interleaved(n,xform);
            std::cout << "transformed to 'interleaved' mcarray " << std::endl;
            xform.print_detailed();
            xform.info(mem_info);
            mem_info.print();
        }
    }

    CONDUIT_INFO("blueprint_example_2");
}

