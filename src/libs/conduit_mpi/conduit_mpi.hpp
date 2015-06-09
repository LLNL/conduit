//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2015, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see https://lc.llnl.gov/conduit/.
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
/// file: conduit_mpi.hpp
///
//-----------------------------------------------------------------------------


#ifndef CONDUIT_MPI_HPP
#define CONDUIT_MPI_HPP

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <mpi.h>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "Conduit_MPI_Exports.hpp"

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{

    struct ConduitMPIRequest {
        MPI_Request _request;
        Node* _externalData;
        Node* _recvData;
    };

//-----------------------------------------------------------------------------
/// Standard MPI Send Recv
//-----------------------------------------------------------------------------

    int CONDUIT_MPI_API send(Node& node,
                             int dest,
                             int tag,
                             MPI_Comm comm);

    int CONDUIT_MPI_API recv(Node& node,
                             int source,
                             int tag,
                             MPI_Comm comm);
// TODO:
// int send_recv(Node &send_node,
//              int dest,
//              int send_tag,
//              Node &recv_node,
//              int source,
//              int recv_tag,
//              MPI_Comm mpi_comm, 
//              MPI_Status *status);

//-----------------------------------------------------------------------------
/// MPI Reduce
//-----------------------------------------------------------------------------
    
    int CONDUIT_MPI_API reduce(Node &send_node,
                               Node &recv_node,
                               MPI_Datatype mpi_datatype,
                               MPI_Op mpi_op,
                               unsigned int root,
                               MPI_Comm comm);

    int CONDUIT_MPI_API all_reduce(Node &send_node,
                                   Node &recv_node,
                                   MPI_Datatype mpi_datatype,
                                   MPI_Op mpi_op,
                                   MPI_Comm comm);

//-----------------------------------------------------------------------------
/// Async MPI Send Recv
//-----------------------------------------------------------------------------

    int CONDUIT_MPI_API isend(Node &node,
                              int dest,
                              int tag,
                              MPI_Comm mpi_comm,
                              ConduitMPIRequest *request);

    int CONDUIT_MPI_API irecv(Node &node,
                              int src,
                              int tag,
                              MPI_Comm comm,
                              ConduitMPIRequest *request);

    int CONDUIT_MPI_API wait_send(ConduitMPIRequest* request,
                                  MPI_Status *status);
   
    int CONDUIT_MPI_API wait_recv(ConduitMPIRequest *request,
                                  MPI_Status *status);

    int CONDUIT_MPI_API wait_all_send(int count,
                                      ConduitMPIRequest requests[],
                                      MPI_Status statuses[]);

    int CONDUIT_MPI_API wait_all_recv(int count,
                                       ConduitMPIRequest requests[],
                                       MPI_Status statuses[]);


// TODO:
//
// int broadcast(Node& node,
//               int root,
//               MPI_Comm comm );
//
// int gather(Node &send_node,
//            Node &recv_node,
//            int root, 
//            MPI_Comm mpi_comm );
//
// int scatter(Node &send_node,
//             Node &recv_node,
//             int root,
//             MPI_Comm mpi_comm );
//
// int all_to_all(Node& send_node,
//                Node& recv_node,
//                MPI_Comm mpi_comm);

//-----------------------------------------------------------------------------
/// The about methods construct human readable info about how conduit_mpi was
/// configured.
//-----------------------------------------------------------------------------
 std::string CONDUIT_MPI_API about();
 void        CONDUIT_MPI_API about(Node &);

};
//-----------------------------------------------------------------------------
// -- end conduit::mpi --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


#endif

