//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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



