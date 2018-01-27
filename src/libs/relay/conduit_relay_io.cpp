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
/// file: conduit_relay_io.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_io.hpp"

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>

// includes for optional features
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
#include "conduit_relay_hdf5.hpp"
#endif

#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
#include "conduit_relay_silo.hpp"
#endif


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

//---------------------------------------------------------------------------//
void
identify_protocol(const std::string &path,
                  std::string &io_type)
{
    io_type = "conduit_bin";

    std::string file_path;
    std::string obj_base;

    // check for ":" split
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    obj_base);

    std::string file_name_base;
    std::string file_name_ext;

    // find file extension to auto match
    conduit::utils::rsplit_string(file_path,
                                  std::string("."),
                                  file_name_ext,
                                  file_name_base);

    
    if(file_name_ext == "hdf5" || 
       file_name_ext == "h5")
    {
        io_type = "hdf5";
    }
    else if(file_name_ext == "silo")
    {
        io_type = "conduit_silo";
    }
    else if(file_name_ext == "json")
    {
        io_type = "json";
    }
    else if(file_name_ext == "conduit_json")
    {
        io_type = "conduit_json";
    }
    else if(file_name_ext == "conduit_base64_json")
    {
        io_type = "conduit_base64_json";
    }
    
    // default to conduit_bin

}

//---------------------------------------------------------------------------//
void 
save(const Node &node,
     const std::string &path)
{
    std::string protocol;
    identify_protocol(path,protocol);
    save(node,path,protocol);
}

//---------------------------------------------------------------------------//
void 
save_merged(const Node &node,
            const std::string &path)
{
    std::string protocol;
    identify_protocol(path,protocol);
    save_merged(node,path,protocol);
}

//---------------------------------------------------------------------------//
void 
load(const std::string &path,
     Node &node)
{
    std::string protocol;
    identify_protocol(path,protocol);
    load(path,protocol,node);
}

//---------------------------------------------------------------------------//
void 
load_merged(const std::string &path,
            Node &node)
{
    std::string protocol;
    identify_protocol(path,protocol);
    load_merged(path,protocol,node);
}


//---------------------------------------------------------------------------//
void 
save(const Node &node,
     const std::string &path,
     const std::string &protocol)
{
    // support conduit::Node's basic save cases
    if(protocol == "conduit_bin" ||
       protocol == "json" || 
       protocol == "conduit_json" ||
       protocol == "conduit_base64_json" )
    {

        node.save(path,protocol);
    }
    else if( protocol == "hdf5")
    {
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
        hdf5_write(node,path);
#else
        CONDUIT_ERROR("conduit_relay lacks HDF5 support: " << 
                      "Failed to save conduit node to path " << path);
#endif
    }
    else if( protocol == "conduit_silo")
    {
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
        silo_write(node,path);
#else
        CONDUIT_ERROR("conduit_relay lacks Silo support: " << 
                      "Failed to save conduit node to path " << path);
#endif
    }
    else if(protocol == "conduit_silo_mesh")
    {
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
        silo_mesh_write(node,path);
#else
        CONDUIT_ERROR("conduit_relay lacks Silo support: " << 
                      "Failed to save conduit mesh node to path " << path);
#endif
    }
    else
    {
        CONDUIT_ERROR("unknown conduit_relay protocol: " << protocol);
    }
}

//---------------------------------------------------------------------------//
void 
save_merged(const Node &node,
            const std::string &path,
            const std::string &protocol)
{
    // support conduit::Node's basic save cases
    if(protocol == "conduit_bin" ||
       protocol == "json" || 
       protocol == "conduit_json" ||
       protocol == "conduit_base64_json" )
    {
        Node n;
        n.load(path,protocol);
        n.update(node);
        n.save(path,protocol);
    }
    else if( protocol == "hdf5")
    {
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
        hdf5_write(node,path);
#else
        CONDUIT_ERROR("conduit_relay lacks HDF5 support: " << 
                      "Failed to save conduit node to path " << path);
#endif
    }
    else if( protocol == "conduit_silo")
    {
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
        Node n;
        silo_read(path,n);
        n.update(node);
        silo_write(n,path);
#else
        CONDUIT_ERROR("conduit_relay lacks Silo support: " << 
                      "Failed to save conduit node to path " << path);
#endif
    }
    else if(protocol == "conduit_silo_mesh")
    {
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
        /// TODO .. ?
        silo_mesh_write(node,path);
#else
        CONDUIT_ERROR("conduit_relay lacks Silo support: " << 
                      "Failed to save conduit mesh node to path " << path);
#endif
    }
    else
    {
        CONDUIT_ERROR("unknown conduit_relay protocol: " << protocol);
    }
}


//---------------------------------------------------------------------------//
void
load(const std::string &path,
     const std::string &protocol,
     Node &node)
{

    // support conduit::Node's basic load cases
    if(protocol == "conduit_bin" ||
       protocol == "json" || 
       protocol == "conduit_json" ||
       protocol == "conduit_base64_json" )
    {
        node.load(path,protocol);
    }
    else if( protocol == "hdf5")
    {
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
        node.reset();
        hdf5_read(path,node);
#else
        CONDUIT_ERROR("conduit_relay lacks HDF5 support: " << 
                      "Failed to load conduit node from path " << path);
#endif
    }
    else if( protocol == "conduit_silo")
    {
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
        silo_read(path,node);
#else
        CONDUIT_ERROR("conduit_relay lacks Silo support: " << 
                    "Failed to load conduit node from path " << path);
#endif
    }
    else if(protocol == "conduit_silo_mesh")
    {
        CONDUIT_ERROR("the conduit_relay conduit_silo_mesh protocol does not "
                      "support \"load\"");
    }
    else
    {
        CONDUIT_ERROR("unknown conduit_relay protocol: " << protocol);
        
    }

}

//---------------------------------------------------------------------------//
void
load_merged(const std::string &path,
            const std::string &protocol,
            Node &node)
{
    // support conduit::Node's basic load cases
    if(protocol == "conduit_bin" ||
       protocol == "json" || 
       protocol == "conduit_json" ||
       protocol == "conduit_base64_json" )
    {
        Node n;
        n.load(path,protocol);
        // update into dest
        node.update(n);

    }
    else if( protocol == "hdf5")
    {
#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
        Node n;
        hdf5_read(path,n);
        node.update(n);
#else
        CONDUIT_ERROR("relay lacks HDF5 support: " << 
                      "Failed to read conduit node from path " << path);
#endif
    }
    else if( protocol == "conduit_silo")
    {
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
        Node n;
        silo_read(path,n);
        node.update(n);
#else
        CONDUIT_ERROR("relay lacks Silo support: " << 
                    "Failed to load conduit node from path " << path);
#endif
    }
    else if(protocol == "conduit_silo_mesh")
    {
        CONDUIT_ERROR("the relay conduit_silo_mesh protocol does not "
                      "support \"load\"");
    }
    else
    {
        CONDUIT_ERROR("relay unknown protocol: " << protocol);
        
    }

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


