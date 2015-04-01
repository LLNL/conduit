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
/// file: conduit_mpi_smoke.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_mpi.h"
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;


TEST(conduit_mpi_smoke, about)
{
    std::cout << mpi::about() << std::endl;
}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_smoke, allreduce) 
{
    Node n1; 
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    n1["value"] = rank*10;

    Node n2;

    mpi::allreduce(n1, n2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    EXPECT_EQ(n2["value"].as_int32(), 10);
}

TEST(conduit_mpi_smoke, isend_irecv_wait) 
{
    Node n1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> doubles;
    

    doubles.push_back(rank+1);
    doubles.push_back(3.4124*rank);
    doubles.push_back(10.7 - rank);

    n1.set_external(doubles);
    


    mpi::ConduitMPIRequest request;

    MPI_Status status;
    if (rank == 0) {
        mpi::Irecv(n1, 1, 0, MPI_COMM_WORLD, &request);
        mpi::Waitrecv(&request, &status);
    } else if (rank == 1) {
        mpi::Isend(n1, 0, 0, MPI_COMM_WORLD, &request);
        mpi::Waitsend(&request, &status);
    }

    EXPECT_EQ(n1.as_float64_ptr()[0], 2);
    EXPECT_EQ(n1.as_float64_ptr()[1], 3.4124);
    EXPECT_EQ(n1.as_float64_ptr()[2], 9.7);
    
}

TEST(conduit_mpi_smoke, waitall) 
{
    Node n1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> doubles;
    

    doubles.push_back(rank+1);
    doubles.push_back(3.4124*rank);
    doubles.push_back(10.7 - rank);

    n1.set_external(doubles);
    


    mpi::ConduitMPIRequest requests[1];

    MPI_Status statuses[1];
    if (rank == 0) {
        mpi::Irecv(n1, 1, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::Waitallrecv(1, requests, statuses);
    } else if (rank == 1) {
        mpi::Isend(n1, 0, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::Waitallsend(1, requests, statuses);
    }

    EXPECT_EQ(n1.as_float64_ptr()[0], 2);
    EXPECT_EQ(n1.as_float64_ptr()[1], 3.4124);
    EXPECT_EQ(n1.as_float64_ptr()[2], 9.7);
    
}

TEST(conduit_mpi_smoke, waitallmultirequest) 
{
    Node n1;
    Node n2;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> doubles;
    

    doubles.push_back(rank+1);
    doubles.push_back(3.4124*rank);
    doubles.push_back(10.7 - rank);

    n1.set_external(doubles);

    n2 = 13123;
    


    mpi::ConduitMPIRequest requests[2];

    MPI_Status statuses[2];
    if (rank == 0) {
        mpi::Irecv(n1, 1, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::Irecv(n2, 1, 0, MPI_COMM_WORLD, &requests[1]);
        mpi::Waitallrecv(1, requests, statuses);
    } else if (rank == 1) {
        mpi::Isend(n1, 0, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::Isend(n2, 0, 0, MPI_COMM_WORLD, &requests[1]);
        mpi::Waitallsend(1, requests, statuses);
    }

    EXPECT_EQ(n1.as_float64_ptr()[0], 2);
    EXPECT_EQ(n1.as_float64_ptr()[1], 3.4124);
    EXPECT_EQ(n1.as_float64_ptr()[2], 9.7);

    EXPECT_EQ(n2.as_int32(), 13123);
    
}

TEST(conduit_mpi_smoke, external) 
{
     Node n1;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<double> doubles1;
    std::vector<double> doubles2;
    std::vector<double> doubles3;
    

    doubles1.push_back(rank+1);
    doubles1.push_back(3.4124*rank);
    doubles1.push_back(10.7 - rank);

    n1.append().set_external(doubles1);

    doubles2.push_back(rank+2);
    doubles2.push_back(3.4124*rank + 1);
    doubles2.push_back(10.7 - rank + 1);

    n1.append().set_external(doubles2);

    doubles3.push_back(rank+3);
    doubles3.push_back(3.4124*rank + 2);
    doubles3.push_back(10.7 - rank + 2);

    n1.append().set_external(doubles3);

    mpi::ConduitMPIRequest request;

    MPI_Status status;
    if (rank == 0) {
        mpi::Irecv(n1, 1, 0, MPI_COMM_WORLD, &request);
        mpi::Waitrecv(&request, &status);
        
    } else if (rank == 1) {
        mpi::Isend(n1, 0, 0, MPI_COMM_WORLD, &request);
        mpi::Waitsend(&request, &status);
    }


    EXPECT_EQ(n1[0].as_float64_ptr()[0], 2);
    EXPECT_EQ(n1[0].as_float64_ptr()[1], 3.4124);
    EXPECT_EQ(n1[0].as_float64_ptr()[2], 9.7);

    EXPECT_EQ(n1[1].as_float64_ptr()[0], 3);
    EXPECT_EQ(n1[1].as_float64_ptr()[1], 4.4124);
    EXPECT_EQ(n1[1].as_float64_ptr()[2], 10.7);
    
    EXPECT_EQ(n1[2].as_float64_ptr()[0], 4);
    EXPECT_EQ(n1[2].as_float64_ptr()[1], 5.4124);
    EXPECT_EQ(n1[2].as_float64_ptr()[2], 11.7);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    result = RUN_ALL_TESTS();
    MPI_Finalize();

    return result;
}
