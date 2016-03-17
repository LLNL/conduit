//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
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
/// file: conduit_io_rest.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_io.hpp"
#include <iostream>
#include "gtest/gtest.h"

#include "t_config.hpp"

using namespace conduit;

bool launch_server = false;
bool use_ssl       = false;
bool use_auth      = false;

TEST(conduit_io_rest, rest_server)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    uint32 c_val = 13;

    Node *n = new Node();
    n->fetch("a") = a_val;
    n->fetch("b") = b_val;
    n->fetch("c") = c_val;

    EXPECT_EQ(n->fetch("a").as_uint32(), a_val);
    EXPECT_EQ(n->fetch("b").as_uint32(), b_val);
    EXPECT_EQ(n->fetch("c").as_uint32(), c_val);
    
    if(launch_server)
    {
        
        std::string cert_file   = std::string("");
        std::string auth_domain = std::string("");
        std::string auth_file   = std::string("");
                  
        if(use_ssl)
        {
            cert_file = utils::join_file_path(CONDUIT_T_SRC_DIR,"conduit_io");
            cert_file = utils::join_file_path(cert_file,"t_ssl_cert.pem");
        }

        if(use_auth)
        {
            auth_domain = "test";
            auth_file = utils::join_file_path(CONDUIT_T_SRC_DIR,"conduit_io");
            auth_file = utils::join_file_path(auth_file,"t_htpasswd.txt");
        }
        

        conduit::io::WebServer *svr = conduit::io::VisualizerServer::serve(n,
                                                                           true,
                                                                           8080,
                                                                           cert_file,
                                                                           auth_domain,
                                                                           auth_file);
        delete svr;
    }
    else
    {
        std::cout << "provide \"launch\" as a command line arg "
                  << "to launch a conduit::Node REST test server at "
                  << "http://localhost:8080" << std::endl;
    }

    delete n;
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);

    for(int i=0; i < argc ; i++)
    {
        std::string arg_str(argv[i]);
        if(arg_str == "launch")
        {
            // actually launch the server
            launch_server = true;
        }
        else if(arg_str == "ssl")
        {
            // test using ssl server cert
            use_ssl = true;
        }
        else if(arg_str == "auth")
        {
            // test using htpasswd auth
            // the user name and password for this example are both "test"
            use_auth = true;
        }
    }

    result = RUN_ALL_TESTS();
    return result;
}


