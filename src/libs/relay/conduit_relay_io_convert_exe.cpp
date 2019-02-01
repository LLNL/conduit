//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_io_convert_exe.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_relay.hpp"
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
#include "conduit_relay_io_hdf5.hpp"
#endif
#include <iostream>
#include <stdlib.h>

using namespace conduit;
using namespace conduit::relay;


//-----------------------------------------------------------------------------
void
usage()
{
    std::cout << "usage: conduit_relay_io_convert {input file} {output file}"
              << std::endl << std::endl 
              << " optional arguments:"
              << std::endl
              << "  --read-proto  {relay protocol string used to read data file}"
              << std::endl
              << "  --write-proto {relay protocol string used to read data file}"
              << "  --opts  {options file}"
              << std::endl << std::endl ;

}

//-----------------------------------------------------------------------------
void
parse_args(int argc,
           char *argv[],
           std::string &input_file,
           std::string &output_file,
           std::string &read_proto,
           std::string &write_proto,
           std::string &opts_file)
{
    for(int i=1; i < argc ; i++)
    {
        std::string arg_str(argv[i]);
        if(arg_str == "--read-protocol")
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

    parse_args(argc,
               argv,
               input_file,
               output_file,
               read_proto,
               write_proto,
               opts_file);

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
    else
    {
        relay::io::load(input_file, read_proto, data);
    }

    if(write_proto.empty())
    {
        relay::io::save(data,output_file);
    }
    else
    {
        relay::io::save(data, output_file, write_proto);
    }

    return 0;
}



