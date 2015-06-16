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
/// file: conduit_mpi.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_mpi.hpp"
#include <iostream>

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


//---------------------------------------------------------------------------//
int 
send(Node &node, int dest, int tag, MPI_Comm comm)
{ 

    Schema schema_c;
    node.schema().compact_to(schema_c);
    std::string schema = schema_c.to_json();
    int schema_len = schema.length() + 1;

    std::vector<uint8> data;
    node.serialize(data);
    int data_len = data.size();


    int intArray[2] = { schema_len, data_len };


    int mpiError = MPI_Send(intArray, 2, MPI_INT, dest, tag, comm);


    //Check errors on return value mpiError here
    if (mpiError == MPI_ERR_COMM) {
    } else if (mpiError == MPI_ERR_COUNT) {
    } else if (mpiError == MPI_ERR_TYPE) {
    } else if (mpiError == MPI_ERR_TAG) {
    } else if (mpiError == MPI_ERR_RANK) {
    }

    mpiError = MPI_Send(const_cast <char*> (schema.c_str()), schema_len, MPI_CHAR, dest, tag, comm);

    if (mpiError == MPI_ERR_COMM) {
    } else if (mpiError == MPI_ERR_COUNT) {
    } else if (mpiError == MPI_ERR_TYPE) {
    } else if (mpiError == MPI_ERR_TAG) {
    } else if (mpiError == MPI_ERR_RANK) {
    }

    return MPI_Send((char*)&data[0], data_len, MPI_CHAR, dest, tag, comm);
}

//---------------------------------------------------------------------------//
int
recv(Node &node, int src, int tag, MPI_Comm comm)
{  
    int intArray[2];
    MPI_Status status;

    int mpiError = MPI_Recv(intArray, 2, MPI_INT, src, tag, comm, &status);

    //Check errors on return value mpiError here
    if (mpiError == MPI_ERR_COMM) {
    } else if (mpiError == MPI_ERR_COUNT) {
    } else if (mpiError == MPI_ERR_TYPE) {
    } else if (mpiError == MPI_ERR_TAG) {
    } else if (mpiError == MPI_ERR_RANK) {
    }

    int schema_len = intArray[0];
    int data_len = intArray[1];

    char schema[schema_len + 1];
    char data[data_len + 1];

    mpiError = MPI_Recv(schema, schema_len, MPI_CHAR, src, tag, comm, &status);

    if (mpiError == MPI_ERR_COMM) {
    } else if (mpiError == MPI_ERR_COUNT) {
    } else if (mpiError == MPI_ERR_TYPE) {
    } else if (mpiError == MPI_ERR_TAG) {
    } else if (mpiError == MPI_ERR_RANK) {
    }

    mpiError = MPI_Recv(data, data_len, MPI_CHAR, src, tag, comm, &status);

    Generator node_gen(schema, data);
    /// gen copy 
    node_gen.walk(node);

    return mpiError;
}
//---------------------------------------------------------------------------//
int 
reduce(Node &send_node,
       Node& recv_node,
       MPI_Datatype mpi_datatype,
       MPI_Op mpi_op,
       unsigned int root,
       MPI_Comm mpi_comm) 
{

    int temp;
    MPI_Comm_rank(MPI_COMM_WORLD, &temp);
    const unsigned int rank = temp;

    Schema schema_c;
    send_node.schema().compact_to(schema_c);
    std::string schema = schema_c.to_json();


    std::vector<uint8> data;
    send_node.serialize(data);
    int data_len = data.size();

    int datasize = 0;
    MPI_Type_size(mpi_datatype, &datasize);

    char recvdata[data_len+1];

    int mpi_error = MPI_Reduce(&data[0],
                               recvdata,
                               (data_len / datasize) + 1,
                               mpi_datatype, mpi_op, root, mpi_comm);

    if (rank == root)
    {
        Generator node_gen(schema, recvdata);

        node_gen.walk(recv_node);
    }

    return mpi_error;
}

//--------------------------------------------------------------------------//
int
all_reduce(Node &send_node,
           Node &recv_node,
           MPI_Datatype mpi_datatype,
           MPI_Op mpi_op,
           MPI_Comm mpi_comm)
{

    Schema schema_c;
    send_node.schema().compact_to(schema_c);
    std::string schema = schema_c.to_json();


    std::vector<uint8> data;
    send_node.serialize(data);
    int data_len = data.size();

    int datasize = 0;
    MPI_Type_size(mpi_datatype, &datasize);

    char recvdata[data_len+1];

    int mpi_error = MPI_Allreduce(&data[0],
                                  recvdata,
                                  (data_len / datasize) + 1,
                                  mpi_datatype,
                                  mpi_op,
                                  mpi_comm);

    Generator node_gen(schema, recvdata);

    node_gen.walk(recv_node);


    return mpi_error;
}

//---------------------------------------------------------------------------//
int
isend(Node &node,
      int dest,
      int tag,
      MPI_Comm mpi_comm,
      ConduitMPIRequest* request) 
{
    request->_externalData = new Node();
    node.compact_to(*(request->_externalData));

    return MPI_Isend((char*)request->_externalData->data_ptr(), 
                     request->_externalData->total_bytes(), 
                     MPI_CHAR, 
                     dest, 
                     tag,
                     mpi_comm,
                     &(request->_request));
}

//---------------------------------------------------------------------------//
int 
irecv(Node &node,
      int src,
      int tag,
      MPI_Comm mpi_comm,
      ConduitMPIRequest *request) 
{
    request->_externalData = new Node();
    node.compact_to(*(request->_externalData));

    request->_recvData = &node;

    return MPI_Irecv((char*)request->_externalData->data_ptr(),
                     request->_externalData->total_bytes(),
                     MPI_CHAR,
                     src,
                     tag,
                     mpi_comm,
                     &(request->_request));
}

//---------------------------------------------------------------------------//
int
wait_send(ConduitMPIRequest* request,
          MPI_Status *status) 
{
    int mpi_error = MPI_Wait(&(request->_request), status);

    delete request->_externalData;
    request->_externalData = 0;

    return mpi_error;
}

//---------------------------------------------------------------------------//
int
wait_recv(ConduitMPIRequest* request,
          MPI_Status *status) 
{
    int mpi_error = MPI_Wait(&(request->_request), status);

    request->_recvData->update(*(request->_externalData));

    delete request->_externalData;
    request->_externalData = 0;

    request->_recvData = 0;
    
    return mpi_error;
}

//---------------------------------------------------------------------------//
int
wait_all_send(int count,
              ConduitMPIRequest requests[],
              MPI_Status statuses[]) 
{
     MPI_Request *justrequests = new MPI_Request[count];
     
     for (int i = 0; i < count; ++i) 
     {
         justrequests[i] = requests[i]._request;
     }
     
     int mpi_error = MPI_Waitall(count, justrequests, statuses);

     for (int i = 0; i < count; ++i)
     {
         requests[i]._request = justrequests[i];
         delete requests[i]._externalData;
         requests[i]._externalData = 0;
     }

     delete [] justrequests;

     return mpi_error; 


}

//---------------------------------------------------------------------------//
int
wait_all_recv(int count,
              ConduitMPIRequest requests[],
              MPI_Status statuses[])
{
     MPI_Request *justrequests = new MPI_Request[count];
     
     for (int i = 0; i < count; ++i)
     {
         justrequests[i] = requests[i]._request;
     }
     
     int mpi_error = MPI_Waitall(count, justrequests, statuses);

     for (int i = 0; i < count; ++i)
     {
         requests[i]._recvData->update(*(requests[i]._externalData));

         requests[i]._request = justrequests[i];
         delete requests[i]._externalData;
         requests[i]._externalData = 0;
     }

     delete [] justrequests;

     return mpi_error; 

}

//---------------------------------------------------------------------------//
int
gatherv(Node &send_node,
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
    int data_len   = n_snd_compact.total_bytes();
    
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

    // TODO: Check for fatal mpi_error
                                
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
        
        NodeIterator itr = n_rcv_sizes.iterator();
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
                             MPI_CHAR,
                             schema_rcv_buff,
                             schema_rcv_counts,
                             schema_rcv_displs,
                             MPI_CHAR,
                             root,
                             mpi_comm);

    // TODO: Check for fatal mpi_error

    // build all schemas from JSON, compact them.
    Schema rcv_schema;
    if( m_rank == root )
    {
        //TODO: should we make it easer to create a compact schema?
        Schema st;
        for(int i=0;i < m_size; i++)
        {
            Schema &s = st.append();
            s.set(&schema_rcv_buff[schema_rcv_displs[i]]);
        }
        
        st.compact_to(rcv_schema);
    }

    
    if( m_rank == root )
    {
        // allocate data to hold the gather result
        recv_node.set(rcv_schema);
        data_rcv_buff = (char*)recv_node.data_ptr();
    }
    
    mpi_error = MPI_Gatherv( n_snd_compact.data_ptr(),
                             data_len,
                             MPI_CHAR,
                             data_rcv_buff,
                             data_rcv_counts,
                             data_rcv_displs,
                             MPI_CHAR,
                             root,
                             mpi_comm);

    // TODO: Check for fatal mpi_error
    return mpi_error;
}

//---------------------------------------------------------------------------//
int
allgatherv(Node &send_node,
           Node &recv_node,
           MPI_Comm mpi_comm)
{
    int mpi_error = 0;
    return mpi_error;
}



//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    mpi::about(n);
    return n.to_pure_json();
}

//---------------------------------------------------------------------------//
void
about(Node &n)
{
    n.reset();
    n["mpi"] = "enabled";
}

};
//-----------------------------------------------------------------------------
// -- end conduit::mpi --
//-----------------------------------------------------------------------------



};
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------


