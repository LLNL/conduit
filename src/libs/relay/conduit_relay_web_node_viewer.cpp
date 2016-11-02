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
/// file: conduit_relay_web_node_viewer_exe.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include <iostream>
#include <stdlib.h>

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
              << "  --address {ip address to bind to (default=127.0.0.1)}" 
              << std::endl
              << "  --port {port number to serve on (default=9000)}" 
              << std::endl
              << "  --protocol {relay protocol string used to read data file}"
              << std::endl
              << "  --doc-root {path to http document root}"
              << std::endl
              << "  --htpasswd {htpasswd file for client authentication}"
              << std::endl
              << "  --cert  {https cert file}"
              << std::endl
              << "  --entangle {use entangle to create htpasswd file}"
              << std::endl
              << "  --gateway {gateway for entangle clients}"
              << std::endl << std::endl ;

}

//-----------------------------------------------------------------------------
void
parse_args(int argc,
           char *argv[],
           std::string &address,
           int &port,
           bool &entangle,
           std::string &doc_root,
           std::string &data_file,
           std::string &protocol,
           std::string &auth_file,
           std::string &cert_file,
           std::string &gateway)
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
        else if(arg_str == "--doc-root")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --doc-root option");
            }

            doc_root = std::string(argv[i+1]);
            i++;
        }
        else if(arg_str == "--address")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --address option");
            }

            address = std::string(argv[i+1]);
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
        else if(arg_str == "--gateway")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --gateway option");
            }

            gateway = std::string(argv[i+1]);
            i++;
        }
        else if(arg_str == "--entangle")
        {
            entangle = true;
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
        bool entangle = false;
        std::string doc_root("");
        std::string data_file("");
        std::string address("127.0.0.1");
        std::string protocol("");
        std::string auth_file("");
        std::string cert_file("");
        std::string gateway("");

        parse_args(argc,
                   argv,
                   address,
                   port,
                   entangle,
                   doc_root,
                   data_file,
                   protocol,
                   auth_file,
                   cert_file,
                   gateway);

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

        // setup our node viewer web server
        web::NodeViewerServer svr;
        
        // provide our data
        svr.set_node(&data);
        
        // set the address
        svr.set_bind_address(address);

        // set the port
        svr.set_port(port);

        // set doc root if passed on command line
        if(!doc_root.empty())
        {
            svr.set_document_root(doc_root);
        }

        // set entangle gateway if passed on command line
        if(!gateway.empty())
        {
            svr.set_entangle_gateway(gateway);
        }

        // set htpasswd file if passed on command line
        if(!auth_file.empty())
        {
            svr.set_htpasswd_auth_file(auth_file);
        }

        // set ssl cert file if passed on command line
        if(!cert_file.empty())
        {
            svr.set_ssl_certificate_file(cert_file);
        }
        
        // run entangle if requested
        if(entangle)
        {
            svr.entangle_register();
        }

        // start the server
        svr.serve(true);

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



