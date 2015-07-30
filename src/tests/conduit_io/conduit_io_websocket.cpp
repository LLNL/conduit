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
// For details, see https://lc.llnl.gov/conduit/.
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

using namespace conduit;
using namespace conduit::utils;
using namespace conduit::io;


bool launch_server = false;

TEST(conduit_io_websocket, websocket_test)
{
    if(! launch_server)
    {
        return;
    }


    
    // read png data into a string.
    // in the real example, we should have the png in memory
    std::string wsock_path = utils::join_file_path(CONDUIT_WEB_CLIENT_ROOT,
                                                   "wsock_test");

    std::string example_png_path = utils::join_file_path(wsock_path,
                                                         "example.png");

    CONDUIT_INFO("png path:" << example_png_path);
    std::ifstream file(example_png_path,
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
        CONDUIT_ERROR("DIDN'T READ ANYTHING");
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
    
    WebServer svr;
    
    // start our server
    svr.serve(wsock_path);

    // this loop won't be necessary in the strawman lib.    
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
            launch_server = true;;
        }
    }

    result = RUN_ALL_TESTS();
    return result;
}


