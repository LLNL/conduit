// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_websocket.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include <iostream>
#include "gtest/gtest.h"

#include "t_config.hpp"

using namespace conduit;
using namespace conduit::utils;
using namespace conduit::relay;


bool launch_server = false;
bool use_ssl       = false;
bool use_auth      = false;

TEST(conduit_relay_web_websocket, websocket_test)
{
    if(! launch_server)
    {
        return;
    }

    // read png data into a string.
    std::string wsock_path = utils::join_file_path(web::web_client_root_directory(),
                                                   "wsock_test");

    std::string example_png_path = utils::join_file_path(wsock_path,
                                                         "example.png");

    CONDUIT_INFO("Reading Example PNG file:" << example_png_path);
    std::ifstream file(example_png_path.c_str(),
                       std::ios::binary);

    // find out how big the png file is
    file.seekg(0, std::ios::end);
    std::streamsize png_raw_bytes = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // use a node to hold the buffers for raw and base64 encoded png data
    Node png_data;
    png_data["raw"].set(DataType::c_char(png_raw_bytes));
    char *png_raw_ptr = png_data["raw"].value();
    
    // read in the raw png data
    if(!file.read(png_raw_ptr, png_raw_bytes))
    {
        // ERROR!
        CONDUIT_ERROR("Failed to read PNG file:" << example_png_path);
    }

    // base64 encode the raw png data
    png_data["encoded"].set(DataType::char8_str(png_raw_bytes*2));
    
    utils::base64_encode(png_raw_ptr,
                         png_raw_bytes,
                         png_data["encoded"].data_ptr());

    // create the message we want to send.
    Node msg;
    msg["type"] = "image";
    // the goal is to drop this directly into a <img> element in our web client
    msg["data"] = "data:image/png;base64," + png_data["encoded"].as_string();
    // we will update the count with every send
    msg["count"] = 0;
    
    msg.to_json_stream("test.json","json");

    // setup the webserver
    web::WebServer svr;

    if(use_ssl)
    {
        std::string cert_file = utils::join_file_path(CONDUIT_T_SRC_DIR,"relay");
        cert_file = utils::join_file_path(cert_file,"t_ssl_cert.pem");
        svr.set_ssl_certificate_file(cert_file);
    }

    if(use_auth)
    {
        std::string auth_file = utils::join_file_path(CONDUIT_T_SRC_DIR,"relay");
        auth_file = utils::join_file_path(auth_file,"t_htpasswd.txt");
        svr.set_htpasswd_auth_domain("test");
        svr.set_htpasswd_auth_file(auth_file);
    }

    svr.set_port(8081);
    svr.set_document_root(wsock_path);

    svr.serve();

    while(svr.is_running()) 
    {
        utils::sleep(1000);
        
        // websocket() returns the first active websocket
        svr.websocket()->send(msg);
        // or with a very short timeout
        //svr.websocket(10,100)->send(msg);
        
        msg["count"] = msg["count"].to_int64() + 1;
    }
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


