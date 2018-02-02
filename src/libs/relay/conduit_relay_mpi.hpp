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

    struct Request
    {
        MPI_Request  m_request;
        Node         m_buffer;
        Node        *m_rcv_ptr;
    };


//-----------------------------------------------------------------------------
/// Helpers for MPI Params
//-----------------------------------------------------------------------------
    int CONDUIT_RELAY_API size(MPI_Comm mpi_comm);
    
    int CONDUIT_RELAY_API rank(MPI_Comm mpi_comm);

//-----------------------------------------------------------------------------
/// Helpers for converting between MPI data types  and conduit data types
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Helpers for converting between mpi dtypes and conduit dtypes
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
    MPI_Datatype CONDUIT_RELAY_API conduit_dtype_to_mpi_dtype(
                                                        const DataType &dtype);

//-----------------------------------------------------------------------------
    index_t CONDUIT_RELAY_API mpi_dtype_to_conduit_dtype_id(MPI_Datatype dt);


//-----------------------------------------------------------------------------
/// Standard MPI Send Recv
//-----------------------------------------------------------------------------

    int CONDUIT_RELAY_API send(const Node &node,
                                int dest,
                                int tag,
                                MPI_Comm comm);

    int CONDUIT_RELAY_API recv(Node &node,
                               int source,
                               int tag,
                               MPI_Comm comm);

    int CONDUIT_RELAY_API send_using_schema(const Node &node,
                                            int dest,
                                            int tag,
                                            MPI_Comm comm);

    int CONDUIT_RELAY_API recv_using_schema(Node &node,
                                            int source,
                                            int tag,
                                            MPI_Comm comm);


//-----------------------------------------------------------------------------
/// MPI Reduce
//-----------------------------------------------------------------------------
    
    /// MPI reduce and all reduce methods. 

    /// While the input does not need to be compact, 
    /// reductions require all MPI ranks have identical compact representations.

    /// These methods do not check across ranks for identical compact 
    /// representation.

    /// Conduit empty, object, and list dtypes can not be reduced.

    /// If the send_node is not compact, it will be compacted prior to sending.

    /// for reduce on the root rank and all_reduce for all ranks:
    ///   if the recv_node is compatible but not compact, data will be placed 
    ///   into a compact buffer, then read back out into the recv_node node. 
    /// 
    ///   if the recv_node is not compatible, it will be reset to
    ///   a compact compatible type.

    int CONDUIT_RELAY_API reduce(const Node &send_node,
                                 Node &recv_node,
                                 MPI_Op mpi_op,
                                 int root,
                                 MPI_Comm comm);

    int CONDUIT_RELAY_API all_reduce(const Node &send_node,
                                     Node &recv_node,
                                     MPI_Op mpi_op,
                                     MPI_Comm comm);


//-----------------------------------------------------------------------------
/// MPI Reduce Helpers
//-----------------------------------------------------------------------------
    
    int CONDUIT_RELAY_API sum_reduce(const Node &send_node,
                                     Node &recv_node,
                                     int root,
                                     MPI_Comm comm);


    int CONDUIT_RELAY_API min_reduce(const Node &send_node,
                                     Node &recv_node,
                                     int root,
                                     MPI_Comm comm);

    int CONDUIT_RELAY_API max_reduce(const Node &send_node,
                                     Node &recv_node,
                                     int root,
                                     MPI_Comm comm);

    int CONDUIT_RELAY_API prod_reduce(const Node &send_node,
                                      Node &recv_node,
                                      int root,
                                      MPI_Comm comm);


    
    int CONDUIT_RELAY_API sum_all_reduce(const Node &send_node,
                                         Node &recv_node,
                                         MPI_Comm comm);


    int CONDUIT_RELAY_API min_all_reduce(const Node &send_node,
                                         Node &recv_node,
                                         MPI_Comm comm);

    int CONDUIT_RELAY_API max_all_reduce(const Node &send_node,
                                         Node &recv_node,
                                         MPI_Comm comm);

    int CONDUIT_RELAY_API prod_all_reduce(const Node &send_node,
                                          Node &recv_node,
                                          MPI_Comm comm);


//-----------------------------------------------------------------------------
/// Async MPI Send Recv
//-----------------------------------------------------------------------------

    int CONDUIT_RELAY_API isend(const Node &node,
                                int dest,
                                int tag,
                                MPI_Comm mpi_comm,
                                Request *request);

    int CONDUIT_RELAY_API irecv(Node &node,
                                int src,
                                int tag,
                                MPI_Comm comm,
                                Request *request);

    int CONDUIT_RELAY_API wait_send(Request *request,
                                    MPI_Status *status);
   
    int CONDUIT_RELAY_API wait_recv(Request *request,
                                    MPI_Status *status);

    int CONDUIT_RELAY_API wait_all_send(int count,
                                        Request requests[],
                                        MPI_Status statuses[]);

    int CONDUIT_RELAY_API wait_all_recv(int count,
                                        Request requests[],
                                        MPI_Status statuses[]);


//-----------------------------------------------------------------------------
/// MPI gather
//-----------------------------------------------------------------------------

    // these expect identical schemas
    int CONDUIT_RELAY_API gather(Node &send_node,
                                 Node &recv_node,
                                 int root,
                                 MPI_Comm mpi_comm);

    int CONDUIT_RELAY_API all_gather(Node &send_node,
                                     Node &recv_node,
                                     MPI_Comm mpi_comm);


    int CONDUIT_RELAY_API gather_using_schema(Node &send_node,
                                              Node &recv_node,
                                              int root, 
                                              MPI_Comm mpi_comm);

    int CONDUIT_RELAY_API all_gather_using_schema(Node &send_node,
                                                  Node &recv_node,
                                                  MPI_Comm mpi_comm);

//-----------------------------------------------------------------------------
/// MPI broadcast
//-----------------------------------------------------------------------------

    int CONDUIT_RELAY_API broadcast(Node &node,
                                    int root,
                                    MPI_Comm comm );

    int CONDUIT_RELAY_API broadcast_using_schema(Node &node,
                                                 int root,
                                                 MPI_Comm comm );

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

