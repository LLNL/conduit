// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_convert_exe.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_relay.hpp"
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
#include "conduit_relay_io_hdf5.hpp"
#endif
#include "conduit_relay_io_blueprint.hpp"
#include <iostream>
#include <stdlib.h>

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
void
usage()
{
    std::cout << "usage: conduit_relay_io_convert {input file} {output file} [--read-protocol protocol] [--write-protocol protocol] [--opts filename] [--help]"
              << std::endl << std::endl 
              << " optional arguments:"
              << std::endl
              << "  --read-protocol  {relay protocol} Protocol used to read data file, e.g. \"hdf5\""
              << std::endl
              << "  --write-protocol {relay protocol} Protocol used to write data file"
              << std::endl
              << "  --opts {options file}             Set options file used to initialize program."
              << std::endl
              << "  --help/-h                         Print usage and exit."
              << std::endl
              << std::endl;

}

//-----------------------------------------------------------------------------
int
parse_args(int argc,
           char *argv[],
           std::string &input_file,
           std::string &output_file,
           std::string &read_proto,
           std::string &write_proto,
           std::string &opts_file)
{
    int retval = 0;
    for(int i=1; i < argc ; i++)
    {
        std::string arg_str(argv[i]);
        if(arg_str == "-h" || arg_str == "--help")
        {
            retval = -1;
            break;
        }
        else if(arg_str == "--read-protocol")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --read-protocol option");
            }

            read_proto = std::string(argv[i+1]);
            i++;
        }
        else if(arg_str == "--write-protocol")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --write-protocol option");
            }

            write_proto = std::string(argv[i+1]);
            i++;
        }
        else if(arg_str == "--opts")
        {
            if(i+1 >= argc )
            {
                CONDUIT_ERROR("expected value following --opts option");
            }

            opts_file = std::string(argv[i+1]);
            i++;
        }
        else if(input_file == "")
        {
            input_file = arg_str;
        }
        else if(output_file == "")
        {
            output_file = arg_str;
        }
    }
    return retval;
}



//-----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
    if(argc < 2)
    {
        usage();
        return -1;
    }
    
    std::string input_file("");
    std::string output_file("");
    std::string read_proto("");
    std::string write_proto("");
    std::string opts_file("");

    if(parse_args(argc,
                  argv,
                  input_file,
                  output_file,
                  read_proto,
                  write_proto,
                  opts_file) < 0)
    {
        usage();
        return -2;
    }

    if(opts_file != "")
    {
        Node opts;
        CONDUIT_INFO("Using opts file:" << opts_file);
        io::load(opts_file,opts);
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
        io::hdf5_set_options(opts["hdf5"]);
#endif
    }
    
    relay::about();
    
    if(input_file.empty())
    {
        CONDUIT_ERROR("no input file passed");
    }

    if(output_file.empty())
    {
        CONDUIT_ERROR("no output file passed");
    }

    // load data from the file
    Node data;
    if(read_proto.empty())
    {
        relay::io::load(input_file,data);
    }
    else if(read_proto == "blueprint")
    {
        // NOTE: "blueprint" is not technically a protocol for Conduit but it
        //        does require a different function to read.
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
        relay::io::blueprint::load_mesh(input_file, data);
#else
        std::cout << "A Blueprint root file cannot be read without HDF5." << std::endl;
#endif
    }
    else
    {
        relay::io::load(input_file, read_proto, data);
    }

    if(write_proto.empty())
    {
        relay::io::save(data,output_file);
    }
    else if(write_proto == "blueprint")
    {
        // NOTE: "blueprint" is not technically a protocol for Conduit but it
        //        does require a different function to write.
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
        relay::io::blueprint::save_mesh(data, output_file, "hdf5");
#else
        std::cout << "A Blueprint root file cannot be written without HDF5." << std::endl;
#endif
    }
    else
    {
        relay::io::save(data, output_file, write_proto);
    }

    return 0;
}
