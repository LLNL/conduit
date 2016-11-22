//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://llnl.github.io/conduit/.
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
/// file: conduit_relay_mpi.hpp
///
//-----------------------------------------------------------------------------


#ifndef CONDUIT_RELAY_MPI_HPP
#define CONDUIT_RELAY_MPI_HPP

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <mpi.h>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_relay_exports.h"

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

    struct ConduitMPIRequest {
        MPI_Request _request;
        Node* _externalData;
        Node* _recvData;
    };


//-----------------------------------------------------------------------------
/// Helpers for MPI Params
//-----------------------------------------------------------------------------
    int CONDUIT_RELAY_API size(MPI_Comm mpi_comm);
    
    int CONDUIT_RELAY_API rank(MPI_Comm mpi_comm);

//-----------------------------------------------------------------------------
/// Standard MPI Send Recv
//-----------------------------------------------------------------------------

    int CONDUIT_RELAY_API send(Node& node,
                               int dest,
                               int tag,
                               MPI_Comm comm);

    int CONDUIT_RELAY_API recv(Node& node,
                               int source,
                               int tag,
                               MPI_Comm comm);
// TODO:
// int CONDUIT_RELAY_API send_recv(Node &send_node,
//                                int dest,
//                                int send_tag,
//                                Node &recv_node,
//                                int source,
//                                int recv_tag,
//                                MPI_Comm mpi_comm, 
//                                MPI_Status *status);

//-----------------------------------------------------------------------------
/// MPI Reduce
//-----------------------------------------------------------------------------
    
    int CONDUIT_RELAY_API reduce(Node &send_node,
                                 Node &recv_node,
                                 MPI_Datatype mpi_datatype,
                                 MPI_Op mpi_op,
                                 int root,
                                 MPI_Comm comm);

    int CONDUIT_RELAY_API all_reduce(Node &send_node,
                                   Node &recv_node,
                                   MPI_Datatype mpi_datatype,
                                   MPI_Op mpi_op,
                                   MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Async MPI Send Recv
//-----------------------------------------------------------------------------

    int CONDUIT_RELAY_API isend(Node &node,
                                int dest,
                                int tag,
                                MPI_Comm mpi_comm,
                                ConduitMPIRequest *request);

    int CONDUIT_RELAY_API irecv(Node &node,
                                int src,
                                int tag,
                                MPI_Comm comm,
                                ConduitMPIRequest *request);

    int CONDUIT_RELAY_API wait_send(ConduitMPIRequest* request,
                                    MPI_Status *status);
   
    int CONDUIT_RELAY_API wait_recv(ConduitMPIRequest *request,
                                    MPI_Status *status);

    int CONDUIT_RELAY_API wait_all_send(int count,
                                        ConduitMPIRequest requests[],
                                        MPI_Status statuses[]);

    int CONDUIT_RELAY_API wait_all_recv(int count,
                                        ConduitMPIRequest requests[],
                                        MPI_Status statuses[]);


//-----------------------------------------------------------------------------
/// MPI gather
//-----------------------------------------------------------------------------

    // the non-v variants expect identical schemas
    int CONDUIT_RELAY_API gather(Node &send_node,
                                 Node &recv_node,
                                 int root,
                                 MPI_Comm mpi_comm);

    int CONDUIT_RELAY_API all_gather(Node &send_node,
                                     Node &recv_node,
                                     MPI_Comm mpi_comm);

    // the v variants work for varying schemas
    int CONDUIT_RELAY_API gatherv(Node &send_node,
                                  Node &recv_node,
                                  int root, 
                                  MPI_Comm mpi_comm);

    int CONDUIT_RELAY_API all_gatherv(Node &send_node,
                                      Node &recv_node,
                                      MPI_Comm mpi_comm);


// TODO:
//
// int CONDUIT_RELAY_API broadcast(Node& node,
//                                int root,
//                                MPI_Comm comm );
//
// int CONDUIT_RELAY_API scatter(Node &send_node,
//                               Node &recv_node,
//                               int root,
//                               MPI_Comm mpi_comm );
//
// int CONDUIT_RELAY_API all_to_all(Node& send_node,
//                                  Node& recv_node,
//                                  MPI_Comm mpi_comm);
//
// int CONDUIT_RELAY_API all_to_allv(Node& send_node,
//                                   Node& recv_node,
//                                   MPI_Comm mpi_comm);


//-----------------------------------------------------------------------------
/// The about methods construct human readable info about how conduit_mpi was
/// configured.
//-----------------------------------------------------------------------------
 std::string CONDUIT_RELAY_API about();
 void        CONDUIT_RELAY_API about(Node &);

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

