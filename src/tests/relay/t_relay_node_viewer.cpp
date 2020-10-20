// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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
            std::string cert_file = utils::join_file_path(CONDUIT_T_SRC_DIR,
                                                          "relay");
            cert_file = utils::join_file_path(cert_file,"t_ssl_cert.pem");
            svr.set_ssl_certificate_file(cert_file);
        }

        if(use_auth)
        {
            std::string auth_file = utils::join_file_path(CONDUIT_T_SRC_DIR,
                                                          "relay");
            auth_file = utils::join_file_path(auth_file,"t_htpasswd.txt");
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


