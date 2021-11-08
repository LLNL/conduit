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
    Node n_about_bp, n_about_relay_io;
    conduit::blueprint::about(n_about_bp);
    conduit::relay::io::about(n_about_relay_io);
    std::cout << "usage:" << std::endl
              << " conduit_blueprint_verify {blueprint-protocol} "
              << "{data file} " << std::endl 
              << " conduit_blueprint_verify {blueprint-protocol} "
              << "{data file} "
              << "--relay-protocol {protocol to use for reading with relay }"
              << std::endl << std::endl 
              << "examples: " << std::endl
              << " conduit_blueprint_verify mesh my_mesh.yaml " << std::endl
              << " conduit_blueprint_verify mesh my_mesh.yaml "
              << "--relay-protocol yaml "<< std::endl
              << " conduit_blueprint_verify mesh my_mesh.root " << std::endl
              << " conduit_blueprint_verify mesh my_mesh.root "
              << "--relay-protocol hdf5 "<< std::endl
              << std::endl << std::endl
              << "[blueprint protocols]" 
              << n_about_bp["protocols"].to_yaml()
              << std::endl << std::endl
              << "[relay protocols]" 
              << n_about_relay_io["protocols"].to_yaml()
              << std::endl << std::endl;


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
        if(argc < 3)
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

        if(!conduit::utils::is_file(data_file))
        {
            CONDUIT_ERROR("file: " << data_file << " not found");
        }

        // load data from the file

        // if the file ends in `.root` assume mesh bp file
        // and use conduit::relay::io::blueprint::load_mesh()

        std::string data_file_name_base;
        std::string data_file_name_ext;
        // find file extension to auto match
        conduit::utils::rsplit_string(data_file,
                                      std::string("."),
                                      data_file_name_ext,
                                      data_file_name_base);

        Node data;
        if(data_file_name_ext == "root")
        {
            relay::io::blueprint::load_mesh(data_file,
                                            data);
        }
        else if(relay_protocol.empty())
        {
            // load guessing protocol
            std::cout << "loading: " 
                      << "  file: "<< data_file << std::endl;
            relay::io::load(data_file,data);
        }
        else
        {
            // load with given protocol
            std::cout << "loading: " 
                      << "  file: "<< data_file << std::endl
                      << "  relay_protocol: " << relay_protocol << std::endl;
            relay::io::load(data_file,relay_protocol,data);
        }

        std::cout << "conduit::blueprint::verify" << std::endl
                  << "  blueprint protocol:" << bp_protocol << std::endl;

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
        std::cout << "ERROR running conduit::blueprint::verify"
                  << std::endl
                  << e.message()
                  << std::endl;
        usage();
        return -1;
    }
    
    return 0;
}



