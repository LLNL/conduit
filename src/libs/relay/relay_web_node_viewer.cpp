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
/// file: relay_web_node_viewer_exe.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "relay.hpp"
#include <iostream>

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
void
usage()
{
    std::cout << "usage: conduit_relay_node_viewer {data file}"
              << std::endl << std::endl 
              << " optional arguments:"
              << std::endl
              << "  --port {port number to bind to on localhost}" 
              << std::endl
              << "  --protocol {relay protocol string used to read data file}"
              << std::endl
              << "  --htpasswd {htpasswd file for client authentication}"
              << std::endl
              << "  --cert  {https cert file}"
              << std::endl << std::endl ;

}

//-----------------------------------------------------------------------------
void
parse_args(int argc,
           char *argv[],
           int &port,
           std::string &data_file,
           std::string &protocol,
           std::string &auth_file,
           std::string &cert_file)
{
    for(int i=1; i < argc ; i++)
    {
        std::string arg_str(argv[i]);
        if(arg_str == "--port")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --port option");
            }

            port = atoi(argv[i+1]);
            i++;

        }
        else if(arg_str == "--protocol")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --protocol option");
            }

            protocol = std::string(argv[i+1]);
            i++;
        }
        else if(arg_str == "--htpasswd")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --htpasswd option");
            }

            auth_file = std::string(argv[i+1]);
            i++;
        }
        else if(arg_str == "--cert")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --cert option");
            }

            cert_file = std::string(argv[i+1]);
            i++;
        }
        else if(data_file == "")
        {
            data_file = arg_str;
        }
    }
}



//-----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
    try
    {
        if(argc == 1)
        {
            usage();
            return -1;
        }
        
        int port = 9000;
        std::string data_file("");
        std::string protocol("");
        std::string auth_file("");
        std::string cert_file("");

        parse_args(argc,
                   argv,
                   port,
                   data_file,
                   protocol,
                   auth_file,
                   cert_file);

        if(data_file.empty())
        {
            CONDUIT_ERROR("no data file passed");
        }

        // load data from the file
        Node data;
        if(protocol.empty())
        {
            relay::io::load(data_file,data);
        }
        else
        {
            relay::io::load(data_file,protocol,data);
        }

        // launch the server
        web::WebServer *svr = web::NodeViewerServer::serve(&data,
                                                           true,
                                                           port,
                                                           cert_file,
                                                           "localhost",
                                                           auth_file);
        delete svr;
    }
    catch(const conduit::Error &e)
    {
        std::cout << "Error launching Conduit Relay Node Viewer Server:"
                  << std::endl
                  << e.message()
                  << std::endl;
        usage();
        return -1;
    }
    
    return 0;
}



