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
/// file: conduit_relay_adios.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_adios.hpp"

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <adios.h>

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{


//-----------------------------------------------------------------------------
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
// Private class used to hold options that control adios i/o params.
// 
// These values are read by about(), and are set by io::adios_set_options()
// 
//
//-----------------------------------------------------------------------------

class ADIOSOptions
{
public:
    long        buffer_size; // maybe compute the buffer size if it does not get set by the user.
    std::string transport;

public:
    ADIOSOptions() : buffer_size(1000000), transport("BP")
    {
    }
    
    //------------------------------------------------------------------------
    void set(const Node &opts)
    {
#if 0
// Look at this for reference on nested options.
        if(opts.has_child("compact_storage"))
        {
            const Node &compact = opts["compact_storage"];
            
            if(compact.has_child("enabled"))
            {
                std::string enabled = compact["enabled"].as_string();
                if(enabled == "false")
                {
                    compact_storage_enabled = false;
                }
                else
                {
                    compact_storage_enabled = true;
                }
            }

            if(compact.has_child("threshold"))
            {
                compact_storage_threshold = compact["threshold"].to_value();
            }
        }
#endif
        if(opts.has_child("buffer_size"))
        {
            buffer_size = opts["buffer_size"].as_long();
        }

        if(opts.has_child("transport"))
        {
            transport = opts["transport"].as_string();
        }
#if 0
        Node tmp;
        about(tmp);
        std::cout << "ADIOS options: " << tmp.to_json() << std::endl;
#endif
    }

    //------------------------------------------------------------------------
    void about(Node &opts)
    {
        opts.reset();

        opts["buffer_size"] = buffer_size;
        opts["transport"] = transport;

        // TODO: query the transports.
        opts["read_only/collective"] = true;
        opts["read_only/transports"] = std::string(
"MPI,"
"POSIX,"
"DATASPACES,"
"PHDF5,"
"MPI_LUSTRE,"
"NC4,"
"MPI_AGGREGATE,"
"FLEXPATH,"
"VAR_MERGE");
    }
};

// default adios i/o settings
static ADIOSOptions adiosOptions;


//-----------------------------------------------------------------------------
void
adios_set_options(const Node &opts)
{
    adiosOptions.set(opts);
}

//-----------------------------------------------------------------------------
void
adios_options(Node &opts)
{
    adiosOptions.about(opts);
}

//-----------------------------------------------------------------------------
void adios_save(const Node &node, const std::string &path)
{
    std::cout << "conduit::relay::io::adios_save(node, path=" << path << ")" << std::endl;
}

//-----------------------------------------------------------------------------
void adios_append(const Node &node, const std::string &path)
{
    std::cout << "conduit::relay::io::adios_append(node, path=" << path << ")" << std::endl;
}

//-----------------------------------------------------------------------------
void adios_load(const std::string &path, Node &node)
{
    std::cout << "conduit::relay::io::adios_load(node, path=" << path << ")" << std::endl;
}


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
