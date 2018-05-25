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
/// file: conduit_relay.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"

#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
#include "conduit_relay_hdf5.hpp"
#endif

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>

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


//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    relay::about(n);
    return n.to_json();
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    n.reset();

    n["web"] = "enabled";
    
    Node conduit_about;
    conduit::about(conduit_about);
    
    std::string install_prefix = conduit_about["install_prefix"].as_string();
    std::string web_root = utils::join_file_path(install_prefix,"share");
    web_root = utils::join_file_path(web_root,"conduit");
    web_root = utils::join_file_path(web_root,"web_clients");
    
    n["web_client_root"] =  web_root;

    Node &io_protos = n["io/protocols"];

    // json io
    io_protos["json"] = "enabled";
    io_protos["conduit_json"] = "enabled";

    // standard binary io
    io_protos["conduit_bin"] = "enabled";

#ifdef CONDUIT_RELAY_IO_HDF5_ENABLED
    // straight hdf5 
    io_protos["hdf5"] = "enabled";
    
    io::hdf5_options(n["io/options/hdf5"]);
#else
    // straight hdf5 
    io_protos["hdf5"] = "disabled";
#endif
    
    // silo
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
    // node is packed into two silo objects
    io_protos["conduit_silo"] = "enabled";
#else
    // node is packed into two silo objects
    io_protos["conduit_silo"] = "disabled";
#endif
    
    // silo mesh aware
#ifdef CONDUIT_RELAY_IO_SILO_ENABLED
    io_protos["conduit_silo_mesh"] = "enabled";
#else
    io_protos["conduit_silo_mesh"] = "disabled";
#endif


#ifdef CONDUIT_RELAY_MPI_ENABLED
    n["mpi"] = "enabled";
#else
    n["mpi"] = "disabled";
#endif


}


}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


