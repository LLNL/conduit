// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_io_adios.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_IO_ADIOS_HPP
#define CONDUIT_RELAY_IO_ADIOS_HPP

//-----------------------------------------------------------------------------
// conduit lib include 
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

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
void CONDUIT_RELAY_API adios_initialize_library();

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_finalize_library();

//-----------------------------------------------------------------------------
/// Write node data to a given path
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.bp:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_save(const Node &node,
                                  const std::string &path);

//-----------------------------------------------------------------------------
/// Write node data to a given path in an existing file.
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.bp:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_save_merged(const Node &node,
                                          const std::string &path);

//-----------------------------------------------------------------------------
/// Add a step of node data to an existing file.
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_add_step(const Node &node,
                                      const std::string &path);

//-----------------------------------------------------------------------------
/// Read adios data from given path into the output node 
/// 
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.bp:/path/inside/adios/file"
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_load(const std::string &path,
                                  Node &node);

//-----------------------------------------------------------------------------
/// Read a given step and domain of adios data from given path into the
//  output node.
/// 
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.bp:/path/inside/adios/file"
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_load(const std::string &path,
                                  int step,
                                  int domain,
                                  Node &node);

//-----------------------------------------------------------------------------
/// Pass a Node to set adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_set_options(const Node &opts);

//-----------------------------------------------------------------------------
/// Get a Node that contains adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_options(Node &opts);

//-----------------------------------------------------------------------------
/// Get a number of steps.
//-----------------------------------------------------------------------------
int  CONDUIT_RELAY_API adios_query_number_of_steps(const std::string &path);

//-----------------------------------------------------------------------------
/// Get a number of domains.
//-----------------------------------------------------------------------------
int  CONDUIT_RELAY_API adios_query_number_of_domains(const std::string &path);

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


#endif



