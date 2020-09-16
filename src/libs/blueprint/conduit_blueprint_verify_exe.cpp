// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_verify_exe.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_relay.hpp"
#include "conduit_blueprint.hpp"
#include <iostream>

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
void
usage()
{
    std::cout << "usage: conduit_blueprint_verify {blueprint-protocol} "
              << "{data file} "
              << std::endl << std::endl 
              << " optional arguments:"
              << std::endl
              << "  --relay-protocol {protocol to use for reading with relay }" 
              << std::endl << std::endl ;

}

//-----------------------------------------------------------------------------
void
parse_args(int argc,
           char *argv[],
           std::string &bp_protocol,
           std::string &data_file,
           std::string &relay_protocol)
{
    bp_protocol = std::string(argv[1]);
    data_file   = std::string(argv[2]);
    
    for(int i=3; i < argc ; i++)
    {
        std::string arg_str(argv[i]);
        if(arg_str == "--relay-protocol")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following "
                              "--relay-protocol option");
            }

            relay_protocol = std::string(argv[i+1]);
            i++;
        }
    }
}



//-----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
    try
    {
        if(argc < 2)
        {
            usage();
            return -1;
        }
        
        std::string bp_protocol("");
        std::string data_file("");
        std::string relay_protocol("");

        parse_args(argc,
                   argv,
                   bp_protocol,
                   data_file,
                   relay_protocol);

        if(data_file.empty())
        {
            CONDUIT_ERROR("no data file passed");
        }

        // load data from the file
        Node data;
        if(relay_protocol.empty())
        {
            relay::io::load(data_file,data);
        }
        else
        {
            relay::io::load(data_file,relay_protocol,data);
        }

        Node info;
        if(conduit::blueprint::verify(bp_protocol,data,info))
        {
            std::cout << "conduit::blueprint::verify succeeded" << std::endl;
        }
        else
        {
            std::cout << "conduit::blueprint::verify FAILED" << std::endl;
        }

        std::cout << "verify info:" << std::endl;
        info.print();

    }
    catch(const conduit::Error &e)
    {
        std::cout << "Error launching running conduit::blueprint::verify"
                  << std::endl
                  << e.message()
                  << std::endl;
        usage();
        return -1;
    }
    
    return 0;
}



