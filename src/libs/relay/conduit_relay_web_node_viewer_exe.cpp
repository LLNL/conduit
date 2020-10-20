// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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



