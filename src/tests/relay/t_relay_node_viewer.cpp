//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
/// file: t_relay_node_viewer.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include <iostream>
#include "gtest/gtest.h"

#include "t_config.hpp"

using namespace conduit;
using namespace conduit::relay;

bool launch_server = false;
bool use_ssl       = false;
bool use_auth      = false;

TEST(conduit_relay_web, node_viewer)
{
    uint32 a_val = 20;
    uint32 b_val = 8;
    
    std::vector<float64> c_vals(5,3.14159);

    Node *n = new Node();
    n->fetch("a") = a_val;
    n->fetch("b") = b_val;
    n->fetch("c") = c_vals;

    EXPECT_EQ(n->fetch("a").as_uint32(), a_val);
    EXPECT_EQ(n->fetch("b").as_uint32(), b_val);
    
    if(launch_server)
    {
        web::NodeViewerServer svr;
        svr.set_port(8080);
        svr.set_node(n);
                  
        if(use_ssl)
        {
            std::string cert_file = utils::join_file_paths(CONDUIT_T_SRC_DIR,
                                                          "relay");
            cert_file = utils::join_file_paths(cert_file,"t_ssl_cert.pem");
            svr.set_ssl_certificate_file(cert_file);
        }

        if(use_auth)
        {
            std::string auth_file = utils::join_file_paths(CONDUIT_T_SRC_DIR,
                                                          "relay");
            auth_file = utils::join_file_paths(auth_file,"t_htpasswd.txt");
            svr.set_htpasswd_auth_domain("test");
            svr.set_htpasswd_auth_file(auth_file);
        }

        svr.serve(true);
    }
    else
    {
        std::cout << "Provide \"launch\" as a command line arg "
                  << "to launch a Conduit Node Viewer server at "
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


