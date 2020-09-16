// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_mpi_io_adios.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_RELAY_MPI_IO_ADIOS_HPP
#define CONDUIT_RELAY_MPI_IO_ADIOS_HPP

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <adios_mpi.h>

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
// -- begin conduit::relay::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_initialize_library(MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_finalize_library(MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Write node data to a given path
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.bp:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_save(const Node &node,
                                  const std::string &path,
                                  MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Write node data to a given path in an existing file.
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.bp:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_save_merged(const Node &node,
                                          const std::string &path,
                                          MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Add a step of node data to an existing file.
///
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.adios:/path/inside/adios/file"
/// 
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_add_step(const Node &node,
                                      const std::string &path,
                                      MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Read adios data from given path into the output node 
/// 
/// This methods supports a file system and adios path, joined using a ":"
///  ex: "/path/on/file/system.bp:/path/inside/adios/file"
///
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_load(const std::string &path,
                                  Node &node,
                                  MPI_Comm comm);

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
                                  Node &node,
                                  MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Pass a Node to set adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_set_options(const Node &opts,
                                         MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Get a Node that contains adios i/o options.
//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API adios_options(Node &opts,
                                     MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Get a number of steps.
//-----------------------------------------------------------------------------
int  CONDUIT_RELAY_API adios_query_number_of_steps(const std::string &path,
                                                   MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Get a number of domains.
//-----------------------------------------------------------------------------
int  CONDUIT_RELAY_API adios_query_number_of_domains(const std::string &path,
                                                     MPI_Comm comm);

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi::io --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi --
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

