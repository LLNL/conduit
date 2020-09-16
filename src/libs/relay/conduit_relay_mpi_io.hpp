// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_mpi_io.hpp
///
//-----------------------------------------------------------------------------


#ifndef CONDUIT_RELAY_MPI_IO_HPP
#define CONDUIT_RELAY_MPI_IO_HPP

//-----------------------------------------------------------------------------
// conduit lib include 
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_relay_exports.h"
#include "conduit_relay_config.h"

#include <mpi.h>

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
/// The about methods construct human readable info about how relay io was
/// configured.
//-----------------------------------------------------------------------------
std::string CONDUIT_RELAY_API about(MPI_Comm comm);
void        CONDUIT_RELAY_API about(conduit::Node &res, MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API initialize(MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API finalize(MPI_Comm comm);

///
/// ``save`` works like a 'set' to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const Node &node,
                            const std::string &path,
                            MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const Node &node,
                            const std::string &path,
                            const std::string &protocol,
                            MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save(const Node &node,
                            const std::string &path,
                            const std::string &protocol,
                            const Node &options,
                            MPI_Comm comm);

///
/// ``save_merged`` works like an update to the file.
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_merged(const Node &node,
                                   const std::string &path,
                                   MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_merged(const Node &node,
                                   const std::string &path,
                                   const std::string &protocol,
                                   MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API save_merged(const Node &node,
                                   const std::string &path,
                                   const std::string &protocol,
                                   const Node &options,
                                   MPI_Comm comm);

///
/// ``add_step`` adds a new time step of data to the file.
///
void CONDUIT_RELAY_API add_step(const Node &node,
                                const std::string &path,
                                MPI_Comm comm);

void CONDUIT_RELAY_API add_step(const Node &node,
                                const std::string &path,
                                const std::string &protocol,
                                MPI_Comm comm);

void CONDUIT_RELAY_API add_step(const Node &node,
                                const std::string &path,
                                const std::string &protocol,
                                const Node &options,
                                MPI_Comm comm);

///
/// ``load`` works like a 'set', the node is reset and then populated
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            Node &node,
                            MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            Node &node,
                            MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            const Node &options,
                            Node &node,
                            MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            int step,
                            int domain,
                            Node &node,
                            MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load(const std::string &path,
                            const std::string &protocol,
                            int step,
                            int domain,
                            const Node &options,
                            Node &node,
                            MPI_Comm comm);

///
/// ``load_merged`` works like an update, for the object case, entries are read
///  into the node. If the node is already in the OBJECT_T role, children are 
///  added
///

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_merged(const std::string &path,
                                   Node &node,
                                   MPI_Comm comm);

//-----------------------------------------------------------------------------
void CONDUIT_RELAY_API load_merged(const std::string &path,
                                   const std::string &protocol,
                                   Node &node,
                                   MPI_Comm comm);

///
/// ``query_number_of_steps`` return the number of steps.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API query_number_of_steps(const std::string &path,
                                            MPI_Comm comm);

///
/// ``query_number_of_domains`` return the number of domains.
///
//-----------------------------------------------------------------------------
int CONDUIT_RELAY_API query_number_of_domains(const std::string &path,
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

