//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014, Lawrence Livermore National Security, LLC.
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

#include "conduit_mpi.h"
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

//---------------------------------------------------------------------------//
int send(Node& node, int dest, int tag, MPI_Comm comm)
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

    return MPI_Send(&data[0], data_len, MPI_CHAR, dest, tag, comm);
}

//---------------------------------------------------------------------------//
int recv(Node& node, int src, int tag, MPI_Comm comm)
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
int reduce(Node& sendNode, Node& recvNode, MPI_Datatype datatype, MPI_Op op, unsigned int root, MPI_Comm comm) {

    int temp;
    MPI_Comm_rank(MPI_COMM_WORLD, &temp);
    const unsigned int rank = temp;

    Schema schema_c;
    sendNode.schema().compact_to(schema_c);
    std::string schema = schema_c.to_json();


    std::vector<uint8> data;
    sendNode.serialize(data);
    int data_len = data.size();

    int datasize = 0;
    MPI_Type_size(datatype, &datasize);

    char recvdata[data_len+1];

    int mpiError = MPI_Reduce(&data[0], recvdata, (data_len / datasize) + 1, datatype, op, root, comm);

    if (rank == root) {
        Generator node_gen(schema, recvdata);

        node_gen.walk(recvNode);
    }

    return mpiError;
}

//--------------------------------------------------------------------------//
int allreduce(Node& sendNode, Node& recvNode, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {

    Schema schema_c;
    sendNode.schema().compact_to(schema_c);
    std::string schema = schema_c.to_json();


    std::vector<uint8> data;
    sendNode.serialize(data);
    int data_len = data.size();

    int datasize = 0;
    MPI_Type_size(datatype, &datasize);

    char recvdata[data_len+1];

    int mpiError = MPI_Allreduce(&data[0], recvdata, (data_len / datasize) + 1, datatype, op, comm);

    Generator node_gen(schema, recvdata);

    node_gen.walk(recvNode);


    return mpiError;
}

//---------------------------------------------------------------------------//
int Isend(Node& node, int dest, int tag, MPI_Comm comm, ConduitMPIRequest* request) { 

    request->_externalData = new Node();
    node.compact_to(*(request->_externalData));

    request->_externalData->print_detailed();
    request->_externalData->schema_pointer()->print();
    request->_externalData->info().print();

    return MPI_Isend(request->_externalData->data_pointer(), request->_externalData->total_bytes(), MPI_CHAR, dest, tag, comm, &(request->_request));
}

//---------------------------------------------------------------------------//

int Irecv(Node& node, int src, int tag, MPI_Comm comm, ConduitMPIRequest* request) {

    request->_externalData = new Node();
    node.compact_to(*(request->_externalData));

    request->_recvData = &node;

    return MPI_Irecv(request->_externalData->data_pointer(), request->_externalData->total_bytes(), MPI_CHAR, src, tag, comm, &(request->_request));
}

//---------------------------------------------------------------------------//

int Waitsend(ConduitMPIRequest* request , MPI_Status *status) {
    int mpiError = MPI_Wait(&(request->_request), status);

    delete request->_externalData;
    request->_externalData = 0;
    
    

    return mpiError;
}

//---------------------------------------------------------------------------//

int Waitrecv(ConduitMPIRequest* request , MPI_Status *status) {
    int mpiError = MPI_Wait(&(request->_request), status);

    request->_recvData->update(*(request->_externalData));

    delete request->_externalData;
    request->_externalData = 0;

    request->_recvData = 0;
    
    

    return mpiError;
}

//---------------------------------------------------------------------------//

int Waitallsend(int count, ConduitMPIRequest requests[], MPI_Status statuses[]) {
     MPI_Request* justrequests = new MPI_Request[count];
     for (int i = 0; i < count; ++i) {
         justrequests[i] = requests[i]._request;
     }
     int mpiError = MPI_Waitall(count, justrequests, statuses);

     for (int i = 0; i < count; ++i) {
         requests[i]._request = justrequests[i];
         delete requests[i]._externalData;
         requests[i]._externalData = 0;
     }

     delete [] justrequests;

     return mpiError; 


}

//---------------------------------------------------------------------------//
int Waitallrecv(int count, ConduitMPIRequest requests[], MPI_Status statuses[]) {
     MPI_Request* justrequests = new MPI_Request[count];
     for (int i = 0; i < count; ++i) {
         justrequests[i] = requests[i]._request;
     }
     int mpiError = MPI_Waitall(count, justrequests, statuses);

     for (int i = 0; i < count; ++i) {
         requests[i]._recvData->update(*(requests[i]._externalData));

         requests[i]._request = justrequests[i];
         delete requests[i]._externalData;
         requests[i]._externalData = 0;
     }

     delete [] justrequests;

     return mpiError; 


}


//---------------------------------------------------------------------------//
std::string
about()
{
    Node n;
    mpi::about(n);
    return n.to_json(true,2);
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


