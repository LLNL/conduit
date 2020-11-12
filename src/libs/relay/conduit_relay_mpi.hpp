// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

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

    // wait for either an isend or irecv request
    int CONDUIT_RELAY_API wait(Request *request,
                               MPI_Status *status);

    int CONDUIT_RELAY_API wait_send(Request *request,
                                    MPI_Status *status);
   
    int CONDUIT_RELAY_API wait_recv(Request *request,
                                    MPI_Status *status);

    // wait for batch of isend and/or irecv requests
    int CONDUIT_RELAY_API wait_all(int count,
                                   Request requests[],
                                   MPI_Status statuses[]);

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

