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

