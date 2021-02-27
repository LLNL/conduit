// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay.hpp"
#include "conduit_relay_io.hpp"

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
    return n.to_yaml();
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

#ifdef CONDUIT_RELAY_ZFP_ENABLED
    n["zfp"] = "enabled";
#else
    n["zfp"] = "disabled";
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


