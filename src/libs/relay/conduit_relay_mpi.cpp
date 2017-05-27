//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2017, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_mpi.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_mpi.hpp"
#include <iostream>

//-----------------------------------------------------------------------------
/// The CONDUIT_CHECK_MPI_ERROR macro is used to check return values for 
/// mpi calls.
//-----------------------------------------------------------------------------
#define CONDUIT_CHECK_MPI_ERROR( check_mpi_err_code )               \
{                                                                   \
    if( check_mpi_err_code  != MPI_SUCCESS)                         \
    {                                                               \
        char check_mpi_err_str_buff[MPI_MAX_ERROR_STRING];          \
        int  check_mpi_err_str_len=0;                               \
        MPI_Error_string( check_mpi_err_code ,                      \
                         check_mpi_err_str_buff,                    \
                         &check_mpi_err_str_len);                   \
                                                                    \
        CONDUIT_ERROR("MPI call failed: \n"                         \
                      << " error code = "                           \
                      <<  check_mpi_err_code  << "\n"               \
                      << " error message = "                        \
                      <<  check_mpi_err_str_buff << "\n");          \
        return  check_mpi_err_code;                                 \
    }                                                               \
}


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
int
size(MPI_Comm mpi_comm)
{
    int res;
    MPI_Comm_size(mpi_comm,&res);
    return res;
};

//-----------------------------------------------------------------------------
int
rank(MPI_Comm mpi_comm)
{
    int res;
    MPI_Comm_rank(mpi_comm,&res);
    return res;
}

//-----------------------------------------------------------------------------
MPI_Datatype
conduit_dtype_to_mpi_dtype(const DataType &dt)
{
    MPI_Datatype res = MPI_DATATYPE_NULL;
    switch(dt.id())
    {
        // signed integer types
        case CONDUIT_NATIVE_CHAR_ID:   res = MPI_CHAR;  break;
        case CONDUIT_NATIVE_SHORT_ID:  res = MPI_SHORT; break;
        case CONDUIT_NATIVE_INT_ID:    res = MPI_INT;   break;
        case CONDUIT_NATIVE_LONG_ID:   res = MPI_LONG;  break;

        #if defined(CONDUIT_USE_LONG_LONG)
        case CONDUIT_NATIVE_LONG_LONG_ID: res = MPI_LONG_LONG; break;
        #endif

        // unsigned integer types 
        case CONDUIT_NATIVE_UNSIGNED_CHAR_ID:  res = MPI_UNSIGNED_CHAR;  break;
        case CONDUIT_NATIVE_UNSIGNED_SHORT_ID: res = MPI_UNSIGNED_SHORT; break;
        case CONDUIT_NATIVE_UNSIGNED_INT_ID:   res = MPI_UNSIGNED;       break;
        case CONDUIT_NATIVE_UNSIGNED_LONG_ID:  res = MPI_UNSIGNED_LONG;  break;
        
        #if defined(CONDUIT_USE_LONG_LONG)
        case CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID:
        {
            res = MPI_UNSIGNED_LONG_LONG;
            break;
        }
        #endif

        // floating point types

        case CONDUIT_NATIVE_FLOAT_ID:  res = MPI_FLOAT;  break;
        case CONDUIT_NATIVE_DOUBLE_ID: res = MPI_DOUBLE; break;

        #if defined(CONDUIT_USE_LONG_DOUBLE)
        case CONDUIT_NATIVE_LONG_DOUBLE_ID: res = MPI_LONG_DOUBLE; break;
        #endif


        case DataType::CHAR8_STR_ID:  res = MPI_CHAR;  break;

    }
    
    return res;
}


//-----------------------------------------------------------------------------
index_t
mpi_dtype_to_conduit_dtype_id(MPI_Datatype dt)
{
    index_t res = DataType::EMPTY_ID;

    switch(dt)
    {
        // string type
        case MPI_CHAR:  res = DataType::CHAR8_STR_ID;  break;

        // signed integer types
        case MPI_SHORT: res = CONDUIT_NATIVE_SHORT_ID; break;
        case MPI_INT:   res = CONDUIT_NATIVE_INT_ID;   break;
        case MPI_LONG:  res = CONDUIT_NATIVE_LONG_ID;  break;

        #if defined(CONDUIT_USE_LONG_LONG)
        case MPI_LONG_LONG: res = CONDUIT_NATIVE_LONG_LONG_ID; break;
        #endif

        // unsigned integer types 
        case MPI_BYTE:           res = CONDUIT_NATIVE_UNSIGNED_CHAR_ID;  break;
        case MPI_UNSIGNED_CHAR:  res = CONDUIT_NATIVE_UNSIGNED_CHAR_ID;  break;
        case MPI_UNSIGNED_SHORT: res = CONDUIT_NATIVE_UNSIGNED_SHORT_ID; break;
        case MPI_UNSIGNED:       res = CONDUIT_NATIVE_UNSIGNED_INT_ID;   break;
        case MPI_UNSIGNED_LONG:  res = CONDUIT_NATIVE_UNSIGNED_LONG_ID;  break;
        
        #if defined(CONDUIT_USE_LONG_LONG)
        case MPI_UNSIGNED_LONG_LONG:
        {
            res = CONDUIT_NATIVE_UNSIGNED_LONG_LONG_ID;
            break;
        }
        #endif

        // floating point types

        case MPI_FLOAT:  res = CONDUIT_NATIVE_FLOAT_ID;  break;
        case MPI_DOUBLE: res = CONDUIT_NATIVE_DOUBLE_ID; break;

        #if defined(CONDUIT_USE_LONG_DOUBLE)
        case MPI_LONG_DOUBLE: res = CONDUIT_NATIVE_LONG_DOUBLE_ID; break;
        #endif

    }

    return res;
}


//---------------------------------------------------------------------------//
int 
send_using_schema(const Node &node, int dest, int tag, MPI_Comm comm)
{ 

    Node snd_compact;
    std::string snd_schema = "";

    const void *snd_ptr = node.contiguous_data_ptr();
    int         snd_data_size = node.total_bytes_compact();
    
    if(snd_ptr != NULL && 
       node.is_compact() )
    {
        snd_schema = node.schema().to_json();
    }
    else
    {
        node.compact_to(snd_compact);
        snd_schema = snd_compact.schema().to_json();
        snd_ptr = snd_compact.data_ptr();
    }
    
    int snd_schema_size = snd_schema.length() + 1;


    int mpi_error = MPI_Send(&snd_schema_size,
                             1,
                             MPI_INT,
                             dest,
                             tag,comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    mpi_error = MPI_Send(const_cast <char*> (snd_schema.c_str()),
                         snd_schema_size,
                         MPI_CHAR,
                         dest,
                         tag,
                         comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    
    mpi_error = MPI_Send(const_cast<void*>(snd_ptr),
                         snd_data_size,
                         MPI_BYTE,
                         dest,
                         tag,
                         comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    return mpi_error;
}


//---------------------------------------------------------------------------//
int
recv_using_schema(Node &node, int src, int tag, MPI_Comm comm)
{  
    int rcv_schema_size = 0;
    MPI_Status status;

    // TODO: use MPI_Probe?

    int mpi_error = MPI_Recv(&rcv_schema_size,
                             1,
                             MPI_INT,
                             src,
                             tag,
                             comm,
                             &status);
                             
    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    Node rcv_buffers;
    rcv_buffers["schema"].set(DataType::char8_str(rcv_schema_size+1));

    char *rcv_schema_ptr = rcv_buffers["schema"].as_char8_str();

    mpi_error = MPI_Recv(rcv_schema_ptr,
                         rcv_schema_size,
                         MPI_CHAR,
                         src,
                         tag,
                         comm,&status);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    Schema rcv_schema;
    
    Generator gen(rcv_schema_ptr);
    gen.walk(rcv_schema);


    bool cpy_out = false;

    // if its not compatible, or compact we will have to 
    // use an update to get the data into the output node
    

    void *rcv_data_ptr  = node.contiguous_data_ptr();
    int   rcv_data_size = node.total_bytes_compact();
    
    if( ! rcv_schema.compatible(node.schema()) ||
        rcv_data_ptr == NULL ||
        ! node.is_compact() )
    {
        rcv_buffers["data"].set(rcv_schema);
        rcv_data_ptr  = rcv_buffers["data"].data_ptr();
        rcv_data_size = rcv_schema.total_bytes_compact();
        cpy_out = true;
    }

    
    mpi_error = MPI_Recv(rcv_data_ptr,
                         rcv_data_size,
                         MPI_BYTE,
                         src,
                         tag,
                         comm,
                         &status);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    
    if(cpy_out)
    {
        node.update(rcv_buffers["data"]);
    }
    
    return mpi_error;
}



//---------------------------------------------------------------------------//
int 
send(const Node &node, int dest, int tag, MPI_Comm comm)
{ 
    // assumes size and type are known on the other end
    
    Node snd_compact;

    const void *snd_ptr   = node.contiguous_data_ptr();;
    int          snd_size = node.total_bytes_compact();;
    
    if( snd_ptr == NULL ||
        ! node.is_compact())
    {
         node.compact_to(snd_compact);
         snd_ptr = snd_compact.data_ptr();
    }

    int mpi_error = MPI_Send(const_cast<void*>(snd_ptr),
                             snd_size,
                             MPI_BYTE,
                             dest,
                             tag,
                             comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    return mpi_error;
}

//---------------------------------------------------------------------------//
int
recv(Node &node, int src, int tag, MPI_Comm comm)
{  

    MPI_Status status;    
    Node rcv_compact;

    bool cpy_out = false;

    const void *rcv_ptr  = node.contiguous_data_ptr();
    int         rcv_size = node.total_bytes_compact();

    if( rcv_ptr == NULL  ||
        ! node.is_compact() )
    {
        // we will need to update into rcv node
        cpy_out = true;
        Schema s_rcv_compact;
        node.schema().compact_to(s_rcv_compact);
        rcv_compact.set_schema(s_rcv_compact);
        rcv_ptr  = rcv_compact.data_ptr();
    }

    int mpi_error = MPI_Recv(const_cast<void*>(rcv_ptr),
                             rcv_size,
                             MPI_BYTE,
                             src,
                             tag,
                             comm,
                             &status);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    
    if(cpy_out)
    {
        node.update(rcv_compact);
    }

    return mpi_error;
}


//---------------------------------------------------------------------------//
int 
reduce(const Node &snd_node,
       Node &rcv_node,
       MPI_Op mpi_op,
       int root,
       MPI_Comm mpi_comm) 
{
    MPI_Datatype mpi_dtype = conduit_dtype_to_mpi_dtype(snd_node.dtype());
    
    if(mpi_dtype == MPI_DATATYPE_NULL)
    {
        CONDUIT_ERROR("Unsupported send DataType for mpi::reduce"
                      << snd_node.dtype().name());
    }
    
    void *snd_ptr = NULL;
    void *rcv_ptr = NULL;
    
    Node snd_compact;
    Node rcv_compact;
    
    //note: we don't have to ask for contig in this case, since
    // we can only reduce leaf types 
    if(snd_node.is_compact())
    {
        snd_ptr = const_cast<void*>(snd_node.data_ptr());
    }
    else
    {
        snd_node.compact_to(snd_compact);
        snd_ptr = snd_compact.data_ptr();
    }

    bool cpy_out = false;
    
    int rank = mpi::rank(mpi_comm);
    
    if( rank == root )
    {
        
        rcv_ptr = rcv_node.contiguous_data_ptr();
    

        if( !snd_node.compatible(rcv_node) ||
            rcv_ptr == NULL ||
            !rcv_node.is_compact() )
        {
            // we will need to update into rcv node
            cpy_out = true;

            Schema s_snd_compact;
            snd_node.schema().compact_to(s_snd_compact);
        
            rcv_compact.set_schema(s_snd_compact);
            rcv_ptr = rcv_compact.data_ptr();
        }
    }

    int num_eles = (int) snd_node.dtype().number_of_elements();

    int mpi_error = MPI_Reduce(snd_ptr,
                               rcv_ptr,
                               num_eles,
                               mpi_dtype,
                               mpi_op,
                               root,
                               mpi_comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    if( rank == root  && cpy_out )
    {
        rcv_node.update(rcv_compact);
    }

    return mpi_error;
}

//--------------------------------------------------------------------------//
int
all_reduce(const Node &snd_node,
           Node &rcv_node,
           MPI_Op mpi_op,
           MPI_Comm mpi_comm)
{
    MPI_Datatype mpi_dtype = conduit_dtype_to_mpi_dtype(snd_node.dtype());
    
    if(mpi_dtype == MPI_DATATYPE_NULL)
    {
        CONDUIT_ERROR("Unsupported send DataType for mpi::all_reduce"
                      << snd_node.dtype().name());
    }

    
    void *snd_ptr = NULL;
    void *rcv_ptr = NULL;
    
    Node snd_compact;
    Node rcv_compact;
    
    //note: we don't have to ask for contig in this case, since
    // we can only reduce leaf types 
    if(snd_node.is_compact())
    {
        snd_ptr = const_cast<void*>(snd_node.data_ptr());
    }
    else
    {
        snd_node.compact_to(snd_compact);
        snd_ptr = snd_compact.data_ptr();
    }

    bool cpy_out = false;
    
    
    rcv_ptr = rcv_node.contiguous_data_ptr();
    

    if( !snd_node.compatible(rcv_node) ||
        rcv_ptr == NULL ||
        !rcv_node.is_compact() )
    {
        // we will need to update into rcv node
        cpy_out = true;

        Schema s_snd_compact;
        snd_node.schema().compact_to(s_snd_compact);
        
        rcv_compact.set_schema(s_snd_compact);
        rcv_ptr = rcv_compact.data_ptr();
    }

    int num_eles = (int) snd_node.dtype().number_of_elements();

    int mpi_error = MPI_Allreduce(snd_ptr,
                                  rcv_ptr,
                                  num_eles,
                                  mpi_dtype,
                                  mpi_op,
                                  mpi_comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    
    if(cpy_out)
    {
        rcv_node.update(rcv_compact);
    }
    

    return mpi_error;
}

//-- reduce helpers -- //

//---------------------------------------------------------------------------//
int 
sum_reduce(const Node &snd_node,
           Node &rcv_node,
           int root,
           MPI_Comm mpi_comm) 
{
    return reduce(snd_node,
                  rcv_node,
                  MPI_SUM,
                  root,
                  mpi_comm);
}


//---------------------------------------------------------------------------//
int 
min_reduce(const Node &snd_node,
           Node &rcv_node,
           int root,
           MPI_Comm mpi_comm) 
{
    return reduce(snd_node,
                  rcv_node,
                  MPI_MIN,
                  root,
                  mpi_comm);
}



//---------------------------------------------------------------------------//
int 
max_reduce(const Node &snd_node,
           Node &rcv_node,
           int root,
           MPI_Comm mpi_comm) 
{
    return reduce(snd_node,
                  rcv_node,
                  MPI_MAX,
                  root,
                  mpi_comm);
}



//---------------------------------------------------------------------------//
int 
prod_reduce(const Node &snd_node,
            Node &rcv_node,
            int root,
            MPI_Comm mpi_comm) 
{
    return reduce(snd_node,
                  rcv_node,
                  MPI_PROD,
                  root,
                  mpi_comm);
}



//--- all reduce helpers -- /
//---------------------------------------------------------------------------//
int 
sum_all_reduce(const Node &snd_node,
               Node &rcv_node,
               MPI_Comm mpi_comm) 
{
    return all_reduce(snd_node,
                      rcv_node,
                      MPI_SUM,
                      mpi_comm);
}


//---------------------------------------------------------------------------//
int 
min_all_reduce(const Node &snd_node,
               Node &rcv_node,
               MPI_Comm mpi_comm) 
{
    return all_reduce(snd_node,
                      rcv_node,
                      MPI_MIN,
                      mpi_comm);

}



//---------------------------------------------------------------------------//
int 
max_all_reduce(const Node &snd_node,
               Node &rcv_node,
               MPI_Comm mpi_comm) 
{
    return all_reduce(snd_node,
                      rcv_node,
                      MPI_MAX,
                      mpi_comm);

}


//---------------------------------------------------------------------------//
int 
prod_all_reduce(const Node &snd_node,
                Node &rcv_node,
                MPI_Comm mpi_comm) 
{
    return all_reduce(snd_node,
                      rcv_node,
                      MPI_PROD,
                      mpi_comm);

}



//---------------------------------------------------------------------------//
int
isend(const Node &node,
      int dest,
      int tag,
      MPI_Comm mpi_comm,
      Request *request) 
{
    
    const void *data_ptr  = node.contiguous_data_ptr();
    int         data_size = node.total_bytes_compact();
    
    if( data_ptr == NULL ||
       !node.is_compact() )
    {
        node.compact_to(request->m_buffer);
        data_ptr  = request->m_buffer.data_ptr();
    }
    
    request->m_rcv_ptr = NULL;

    int mpi_error =  MPI_Isend(const_cast<void*>(data_ptr), 
                               data_size, 
                               MPI_BYTE, 
                               dest, 
                               tag,
                               mpi_comm,
                               &(request->m_request));
                               
    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    return mpi_error;
}

//---------------------------------------------------------------------------//
int 
irecv(Node &node,
      int src,
      int tag,
      MPI_Comm mpi_comm,
      Request *request) 
{
    
    // if rcv is compact, we can write directly into recv
    // if its not compact, we need a recv_buffer
    
    void *data_ptr  = node.contiguous_data_ptr();
    int   data_size = node.total_bytes_compact();
    
    if(data_ptr != NULL && 
       node.is_compact() )
    {
        request->m_rcv_ptr = NULL;
    }
    else
    {
        node.compact_to(request->m_buffer);
        data_ptr  = request->m_buffer.data_ptr();
        request->m_rcv_ptr = &node;
    }

    int mpi_error =  MPI_Irecv(data_ptr,
                               data_size,
                               MPI_BYTE,
                               src,
                               tag,
                               mpi_comm,
                               &(request->m_request));

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    return mpi_error;
}

//---------------------------------------------------------------------------//
int
wait_send(Request *request,
          MPI_Status *status) 
{
    int mpi_error = MPI_Wait(&(request->m_request), status);
    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    
    request->m_buffer.reset();

    return mpi_error;
}

//---------------------------------------------------------------------------//
int
wait_recv(Request *request,
          MPI_Status *status) 
{
    int mpi_error = MPI_Wait(&(request->m_request), status);
    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    
    // we need to update if m_rcv_ptr was used
    if(request->m_rcv_ptr)
    {
        request->m_rcv_ptr->update(request->m_buffer);
    }

    request->m_buffer.reset();
    request->m_rcv_ptr = NULL;
    
    return mpi_error;
}

//---------------------------------------------------------------------------//
int
wait_all_send(int count,
              Request requests[],
              MPI_Status statuses[]) 
{
     MPI_Request *justrequests = new MPI_Request[count];
     
     for (int i = 0; i < count; ++i) 
     {
         // mpi requests can be simply copied
         justrequests[i] = requests[i].m_request;
     }
     
     int mpi_error = MPI_Waitall(count, justrequests, statuses);
     CONDUIT_CHECK_MPI_ERROR(mpi_error);

     for (int i = 0; i < count; ++i)
     {
         requests[i].m_request = justrequests[i];
         requests[i].m_buffer.reset();
     }

     delete [] justrequests;

     return mpi_error; 


}

//---------------------------------------------------------------------------//
int
wait_all_recv(int count,
              Request requests[],
              MPI_Status statuses[])
{
     MPI_Request *justrequests = new MPI_Request[count];
     
     for (int i = 0; i < count; ++i)
     {
         // mpi requests can be simply copied
         justrequests[i] = requests[i].m_request;
     }
     
     int mpi_error = MPI_Waitall(count, justrequests, statuses);
     CONDUIT_CHECK_MPI_ERROR(mpi_error);

     for (int i = 0; i < count; ++i)
     {
         if(requests[i].m_rcv_ptr)
         {
             requests[i].m_rcv_ptr->update(requests[i].m_buffer);
             requests[i].m_rcv_ptr = NULL;
         }

         requests[i].m_request = justrequests[i];
         requests[i].m_buffer.reset();
     }

     delete [] justrequests;

     return mpi_error; 

}

//---------------------------------------------------------------------------//
int
gather(Node &send_node,
       Node &recv_node,
       int root,
       MPI_Comm mpi_comm)
{
    Node   n_snd_compact;
    Schema s_snd_compact;
    
    send_node.schema().compact_to(s_snd_compact);
    
    const void *snd_ptr = send_node.contiguous_data_ptr();
    int   snd_size = 0;
    
    
    if(snd_ptr != NULL && 
       send_node.is_compact() )
    {
        snd_ptr  = send_node.data_ptr();
        snd_size = send_node.total_bytes_compact();
    }
    else
    {
        send_node.compact_to(n_snd_compact);
        snd_ptr  = n_snd_compact.data_ptr();
        snd_size = n_snd_compact.total_bytes_compact();
    }

    int mpi_rank = mpi::rank(mpi_comm);
    int mpi_size = mpi::size(mpi_comm);

    if(mpi_rank == root)
    {
        // TODO: copy out support w/o always reallocing?
        recv_node.list_of(s_snd_compact,
                          mpi_size);
    }

    int mpi_error = MPI_Gather( const_cast<void*>(snd_ptr), // local data
                                snd_size, // local data len
                                MPI_BYTE, // send chars
                                recv_node.data_ptr(),  // rcv buffer
                                snd_size, // data len 
                                MPI_BYTE,  // rcv chars
                                root,
                                mpi_comm); // mpi com

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    return mpi_error;
}

//---------------------------------------------------------------------------//
int
all_gather(Node &send_node,
           Node &recv_node,
           MPI_Comm mpi_comm)
{
    Node   n_snd_compact;
    Schema s_snd_compact;
    
    send_node.schema().compact_to(s_snd_compact);
    
    const void *snd_ptr  = send_node.contiguous_data_ptr();
    int         snd_size = send_node.total_bytes_compact();
    
    if( snd_ptr == NULL ||
       !send_node.is_compact() )
    {
        send_node.compact_to(n_snd_compact);
        snd_ptr  = n_snd_compact.data_ptr();
    }


    int mpi_size = mpi::size(mpi_comm);


    // TODO: copy out support w/o always reallocing?
    // TODO: what about common case of scatter w/ leaf types?
    //       instead of list_of, we would have a leaf of
    //       of a given type w/ # of elements == # of ranks. 
    recv_node.list_of(n_snd_compact.schema(),
                      mpi_size);

    int mpi_error = MPI_Allgather( const_cast<void*>(snd_ptr), // local data
                                   snd_size, // local data len
                                   MPI_BYTE, // send chars
                                   recv_node.data_ptr(),  // rcv buffer
                                   snd_size, // data len 
                                   MPI_BYTE,  // rcv chars
                                   mpi_comm); // mpi com

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    return mpi_error;
}



//---------------------------------------------------------------------------//
int
gather_using_schemas(Node &send_node,
                     Node &recv_node,
                     int root, 
                     MPI_Comm mpi_comm)
{
    Node n_snd_compact;
    send_node.compact_to(n_snd_compact);

    int m_size = mpi::size(mpi_comm);
    int m_rank = mpi::rank(mpi_comm);

    std::string schema_str = n_snd_compact.schema().to_json();

    int schema_len = schema_str.length() + 1;
    int data_len   = n_snd_compact.total_bytes_compact();
    
    // to do the conduit gatherv, first need a gather to get the 
    // schema and data buffer sizes
    
    int snd_sizes[] = {schema_len, data_len};

    Node n_rcv_sizes;

    if( m_rank == root )
    {
        Schema s;
        s["schema_len"].set(DataType::c_int());
        s["data_len"].set(DataType::c_int());
        n_rcv_sizes.list_of(s,m_size);
    }

    int mpi_error = MPI_Gather( snd_sizes, // local data
                                2, // two ints per rank
                                MPI_INT, // send ints
                                n_rcv_sizes.data_ptr(),  // rcv buffer
                                2,  // two ints per rank
                                MPI_INT,  // rcv ints
                                root,  // id of root for gather op
                                mpi_comm); // mpi com

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
                                
    Node n_rcv_tmp;
    
    int  *schema_rcv_counts = NULL;
    int  *schema_rcv_displs = NULL;
    char *schema_rcv_buff   = NULL;

    int  *data_rcv_counts = NULL;
    int  *data_rcv_displs = NULL;
    char *data_rcv_buff   = NULL;

    // we only need rcv params on the gather root
    if( m_rank == root )
    {
        // alloc data for the mpi gather counts and displ arrays
        n_rcv_tmp["schemas/counts"].set(DataType::c_int(m_size));
        n_rcv_tmp["schemas/displs"].set(DataType::c_int(m_size));

        n_rcv_tmp["data/counts"].set(DataType::c_int(m_size));
        n_rcv_tmp["data/displs"].set(DataType::c_int(m_size));

        // get pointers to counts and displs
        schema_rcv_counts = n_rcv_tmp["schemas/counts"].value();
        schema_rcv_displs = n_rcv_tmp["schemas/displs"].value();

        data_rcv_counts = n_rcv_tmp["data/counts"].value();
        data_rcv_displs = n_rcv_tmp["data/displs"].value();

        int schema_curr_displ = 0;
        int data_curr_displ   = 0;
        int i=0;
        
        NodeIterator itr = n_rcv_sizes.children();
        while(itr.has_next())
        {
            Node &curr = itr.next();

            int schema_curr_count = curr["schema_len"].value();
            int data_curr_count   = curr["data_len"].value();
            
            schema_rcv_counts[i] = schema_curr_count;
            schema_rcv_displs[i] = schema_curr_displ;
            schema_curr_displ   += schema_curr_count;
            
            data_rcv_counts[i] = data_curr_count;
            data_rcv_displs[i] = data_curr_displ;
            data_curr_displ   += data_curr_count;
            
            i++;
        }
        
        n_rcv_tmp["schemas/data"].set(DataType::c_char(schema_curr_displ));
        schema_rcv_buff = n_rcv_tmp["schemas/data"].value();
    }

    mpi_error = MPI_Gatherv( const_cast <char*>(schema_str.c_str()),
                             schema_len,
                             MPI_BYTE,
                             schema_rcv_buff,
                             schema_rcv_counts,
                             schema_rcv_displs,
                             MPI_BYTE,
                             root,
                             mpi_comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    // build all schemas from JSON, compact them.
    Schema rcv_schema;
    if( m_rank == root )
    {
        //TODO: should we make it easer to create a compact schema?
        // TODO: Revisit, I think we can do this better

        Schema s_tmp;
        for(int i=0;i < m_size; i++)
        {
            Schema &s = s_tmp.append();
            s.set(&schema_rcv_buff[schema_rcv_displs[i]]);
        }
        
        s_tmp.compact_to(rcv_schema);
    }

    
    if( m_rank == root )
    {
        // allocate data to hold the gather result
        // TODO can we support copy out w/out realloc
        recv_node.set(rcv_schema);
        data_rcv_buff = (char*)recv_node.data_ptr();
    }
    
    mpi_error = MPI_Gatherv( n_snd_compact.data_ptr(),
                             data_len,
                             MPI_BYTE,
                             data_rcv_buff,
                             data_rcv_counts,
                             data_rcv_displs,
                             MPI_BYTE,
                             root,
                             mpi_comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    return mpi_error;
}

//---------------------------------------------------------------------------//
int
all_gather_using_schemas(Node &send_node,
                         Node &recv_node,
                         MPI_Comm mpi_comm)
{
    Node n_snd_compact;
    send_node.compact_to(n_snd_compact);

    int m_size = mpi::size(mpi_comm);

    std::string schema_str = n_snd_compact.schema().to_json();

    int schema_len = schema_str.length() + 1;
    int data_len   = n_snd_compact.total_bytes_compact();
    
    // to do the conduit gatherv, first need a gather to get the 
    // schema and data buffer sizes
    
    int snd_sizes[] = {schema_len, data_len};

    Node n_rcv_sizes;

    Schema s;
    s["schema_len"].set(DataType::c_int());
    s["data_len"].set(DataType::c_int());
    n_rcv_sizes.list_of(s,m_size);

    int mpi_error = MPI_Allgather( snd_sizes, // local data
                                   2, // two ints per rank
                                   MPI_INT, // send ints
                                   n_rcv_sizes.data_ptr(),  // rcv buffer
                                   2,  // two ints per rank
                                   MPI_INT,  // rcv ints
                                   mpi_comm); // mpi com

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
                                
    Node n_rcv_tmp;
    
    int  *schema_rcv_counts = NULL;
    int  *schema_rcv_displs = NULL;
    char *schema_rcv_buff   = NULL;

    int  *data_rcv_counts = NULL;
    int  *data_rcv_displs = NULL;
    char *data_rcv_buff   = NULL;


    // alloc data for the mpi gather counts and displ arrays
    n_rcv_tmp["schemas/counts"].set(DataType::c_int(m_size));
    n_rcv_tmp["schemas/displs"].set(DataType::c_int(m_size));

    n_rcv_tmp["data/counts"].set(DataType::c_int(m_size));
    n_rcv_tmp["data/displs"].set(DataType::c_int(m_size));

    // get pointers to counts and displs
    schema_rcv_counts = n_rcv_tmp["schemas/counts"].value();
    schema_rcv_displs = n_rcv_tmp["schemas/displs"].value();

    data_rcv_counts = n_rcv_tmp["data/counts"].value();
    data_rcv_displs = n_rcv_tmp["data/displs"].value();

    int schema_curr_displ = 0;
    int data_curr_displ   = 0;
    int i=0;
    
    NodeIterator itr = n_rcv_sizes.children();
    while(itr.has_next())
    {
        Node &curr = itr.next();

        int schema_curr_count = curr["schema_len"].value();
        int data_curr_count   = curr["data_len"].value();
        
        schema_rcv_counts[i] = schema_curr_count;
        schema_rcv_displs[i] = schema_curr_displ;
        schema_curr_displ   += schema_curr_count;
        
        data_rcv_counts[i] = data_curr_count;
        data_rcv_displs[i] = data_curr_displ;
        data_curr_displ   += data_curr_count;
        
        i++;
    }
    
    n_rcv_tmp["schemas/data"].set(DataType::c_char(schema_curr_displ));
    schema_rcv_buff = n_rcv_tmp["schemas/data"].value();

    mpi_error = MPI_Allgatherv( const_cast <char*>(schema_str.c_str()),
                                schema_len,
                                MPI_BYTE,
                                schema_rcv_buff,
                                schema_rcv_counts,
                                schema_rcv_displs,
                                MPI_BYTE,
                                mpi_comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    // build all schemas from JSON, compact them.
    Schema rcv_schema;
    //TODO: should we make it easer to create a compact schema?
    // TODO: Revisit, I think we can do this better
    Schema s_tmp;
    for(int i=0;i < m_size; i++)
    {
        Schema &s = s_tmp.append();
        s.set(&schema_rcv_buff[schema_rcv_displs[i]]);
    }
    
    // TODO can we support copy out w/out realloc
    s_tmp.compact_to(rcv_schema);

    // allocate data to hold the gather result
    recv_node.set(rcv_schema);
    data_rcv_buff = (char*)recv_node.data_ptr();
    
    mpi_error = MPI_Allgatherv( n_snd_compact.data_ptr(),
                                data_len,
                                MPI_BYTE,
                                data_rcv_buff,
                                data_rcv_counts,
                                data_rcv_displs,
                                MPI_BYTE,
                                mpi_comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    return mpi_error;
}


//---------------------------------------------------------------------------//
int
broadcast(Node &node,
          int root,
          MPI_Comm comm)
{
    int rank = mpi::rank(comm);

    Node bcast_buffer;

    bool cpy_out = false;

    void *bcast_data_ptr  = node.contiguous_data_ptr();
    int   bcast_data_size = node.total_bytes_compact();
        

    // setup buffers on root for send
    if(rank == root)
    {
        if( bcast_data_ptr == NULL ||
            ! node.is_compact() )
        {
            node.compact_to(bcast_buffer);
            bcast_data_ptr  = bcast_buffer.data_ptr();
        }
    
    }
    else // rank != root,  setup buffers on non root for rcv
    {
        if( bcast_data_ptr == NULL ||
            ! node.is_compact() )
        {
            Schema s_compact;
            node.schema().compact_to(s_compact);
            bcast_buffer.set_schema(s_compact);
            
            bcast_data_ptr  = bcast_buffer.data_ptr();
            cpy_out = true;
        }
    }


    int mpi_error = MPI_Bcast(bcast_data_ptr,
                              bcast_data_size,
                              MPI_BYTE,
                              root,
                              comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    // note: cpy_out will always be false when rank == root
    if( cpy_out )
    {
        node.update(bcast_buffer);
    }

    return mpi_error;
}

//---------------------------------------------------------------------------//
int
broadcast_using_schema(Node &node,
                       int root,
                       MPI_Comm comm)
{
    int rank = mpi::rank(comm);

    Node bcast_buffers;

    void *bcast_data_ptr = NULL;
    int   bcast_data_size = 0;

    int bcast_schema_size = 0;
    int rcv_bcast_schema_size = 0;

    // setup buffers for send
    if(rank == root)
    {
        
        bcast_data_ptr  = node.contiguous_data_ptr();
        bcast_data_size = node.total_bytes_compact();
        
        if(bcast_data_ptr != NULL &&
           node.is_compact() )
        {
            bcast_buffers["schema"] = node.schema().to_json();
        }
        else
        {
            Node &bcast_data_compact = bcast_buffers["data"];
            node.compact_to(bcast_data_compact);
            
            bcast_data_ptr  = bcast_data_compact.data_ptr();
            bcast_buffers["schema"] =  bcast_data_compact.schema().to_json();
        }
     

        
        bcast_schema_size = bcast_buffers["schema"].dtype().number_of_elements();
    }

    int mpi_error = MPI_Allreduce(&bcast_schema_size,
                                  &rcv_bcast_schema_size,
                                  1,
                                  MPI_INT,
                                  MPI_MAX,
                                  comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    bcast_schema_size = rcv_bcast_schema_size;

    // alloc for rcv for schema
    if(rank != root)
    {
        bcast_buffers["schema"].set(DataType::char8_str(bcast_schema_size));
    }

    // broadcast the schema 
    mpi_error = MPI_Bcast(bcast_buffers["schema"].data_ptr(),
                          bcast_schema_size,
                          MPI_CHAR,
                          root,
                          comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);
    
    bool cpy_out = false;
    
    // setup buffers for receive 
    if(rank != root)
    {
        Schema bcast_schema;
        Generator gen(bcast_buffers["schema"].as_char8_str());
        gen.walk(bcast_schema);
        
        if( bcast_schema.compatible(node.schema()))
        {
            
            bcast_data_ptr  = node.contiguous_data_ptr();
            bcast_data_size = node.total_bytes_compact();
            
            if( bcast_data_ptr == NULL ||
                ! node.is_compact() )
            {
                Node &bcast_data_buffer = bcast_buffers["data"];
                bcast_data_buffer.set_schema(bcast_schema);
                
                bcast_data_ptr  = bcast_data_buffer.data_ptr();
                cpy_out = true;
            }
        }
        else
        {
            node.set_schema(bcast_schema);

            bcast_data_ptr  = node.data_ptr();
            bcast_data_size = node.total_bytes_compact();
        }
    }
    
    mpi_error = MPI_Bcast(bcast_data_ptr,
                          bcast_data_size,
                          MPI_BYTE,
                          root,
                          comm);

    CONDUIT_CHECK_MPI_ERROR(mpi_error);

    // note: cpy_out will always be false when rank == root
    if( cpy_out )
    {
        node.update(bcast_buffers["data"]);
    }

    return mpi_error;
}


//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    mpi::about(n);
    return n.to_json();
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    n.reset();
    n["mpi"] = "enabled";
}

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


