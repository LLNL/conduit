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
/// file: t_relay_mpi_test.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_mpi.hpp"
#include <iostream>
#include "math.h"
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;
using namespace conduit::relay::mpi;


//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, conduit_dtype_to_mpi_dtype) 
{
    MPI_Datatype dt_test = MPI_DATATYPE_NULL;
    
    // all support types should be mapped (eg: not MPI_DATATYPE_NULL)


    // signed integers
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::int8()),MPI_DATATYPE_NULL);
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::int16()),MPI_DATATYPE_NULL);
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::int32()),MPI_DATATYPE_NULL);
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::int64()),MPI_DATATYPE_NULL);
    
    // unsigned integers
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::uint8()),MPI_DATATYPE_NULL);

    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::uint16()),
              MPI_DATATYPE_NULL);   

    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::uint32()),
              MPI_DATATYPE_NULL);

    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::uint64()),
              MPI_DATATYPE_NULL);

    // floating point
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::float32()),
              MPI_DATATYPE_NULL);
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::float64()),
              MPI_DATATYPE_NULL);

    // string
    EXPECT_NE(conduit_dtype_to_mpi_dtype(DataType::char8_str()),
              MPI_DATATYPE_NULL);

    // empty, object, and list should return null
    EXPECT_EQ(conduit_dtype_to_mpi_dtype(DataType::empty()),MPI_DATATYPE_NULL);
    EXPECT_EQ(conduit_dtype_to_mpi_dtype(DataType::object()),
              MPI_DATATYPE_NULL);
    EXPECT_EQ(conduit_dtype_to_mpi_dtype(DataType::object()),
              MPI_DATATYPE_NULL);

}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, mpi_dtype_to_conduit_dtype) 
{
    EXPECT_TRUE(DataType(mpi_dtype_to_conduit_dtype_id(MPI_INT)).is_int());
    EXPECT_TRUE(DataType(mpi_dtype_to_conduit_dtype_id(MPI_UNSIGNED)).is_unsigned_int());

    EXPECT_TRUE(DataType(mpi_dtype_to_conduit_dtype_id(MPI_FLOAT)).is_float());
    EXPECT_TRUE(DataType(mpi_dtype_to_conduit_dtype_id(MPI_DOUBLE)).is_double());
}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, reduce) 
{
    Node n_snd, n_reduce;
    
    int rank = mpi::rank(MPI_COMM_WORLD);
    int com_size = mpi::size(MPI_COMM_WORLD);
    
    int root = 0;

    int val = rank * 10;
    n_snd = val;

    mpi::reduce(n_snd, n_reduce, MPI_MAX, root, MPI_COMM_WORLD);
    
    if(rank == root)
    {
        EXPECT_EQ(n_reduce.as_int(), 10 * (com_size-1));
    }

    // check non-compact case
    
    n_snd.set(DataType::c_int(2,0,sizeof(int)*2));
    n_reduce.set(DataType::c_int(2,0,sizeof(int)*2));
    
    int_array snd_vals = n_snd.value();
    
    snd_vals[0] = 10;
    snd_vals[1] = 20;
            
    void *reduce_ptr = n_reduce.data_ptr();
    mpi::reduce(n_snd, n_reduce, MPI_SUM, root, MPI_COMM_WORLD);
    
    if(rank == root)
    {
        int_array reduce_vals = n_reduce.value();
    
        EXPECT_EQ(reduce_vals[0], 10 * com_size);
        EXPECT_EQ(reduce_vals[1], 20 * com_size);
    
        // n_reduce should have the same data pointer before and after the reduce
        EXPECT_EQ(reduce_ptr,n_reduce.data_ptr());
    }

    // set to something in-compat
    n_reduce.set(3.1415);
    
    n_reduce.print();

    Schema s_reduce_pre;
    
    s_reduce_pre =  n_reduce.schema();
    
    mpi::reduce(n_snd, n_reduce, MPI_SUM, root, MPI_COMM_WORLD);
    
    if(rank == root)
    {
        n_reduce.print();
    
        int_array reduce_vals = n_reduce.value();
        
        EXPECT_EQ(reduce_vals[0], 10 * com_size);
        EXPECT_EQ(reduce_vals[1], 20 * com_size);

        EXPECT_FALSE(s_reduce_pre.compatible(n_reduce.schema()));
    }

}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, allreduce) 
{
    Node n_snd, n_reduce;
    
    int rank = mpi::rank(MPI_COMM_WORLD);
    int com_size = mpi::size(MPI_COMM_WORLD);

    int val = rank * 10;
    n_snd = val;

    mpi::all_reduce(n_snd, n_reduce, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(n_reduce.as_int(), 10 * (com_size-1));

    // check non-compact case
    
    n_snd.set(DataType::c_int(2,0,sizeof(int)*2));
    n_reduce.set(DataType::c_int(2,0,sizeof(int)*2));
    
    int_array snd_vals = n_snd.value();
    
    snd_vals[0] = 10;
    snd_vals[1] = 20;
            
    void *reduce_ptr = n_reduce.data_ptr();
    mpi::all_reduce(n_snd, n_reduce, MPI_SUM, MPI_COMM_WORLD);
    
    
    int_array reduce_vals = n_reduce.value();
    
    EXPECT_EQ(reduce_vals[0], 10 * com_size);
    EXPECT_EQ(reduce_vals[1], 20 * com_size);
    
    // n_reduce should have the same data pointer before and after the reduce
    EXPECT_EQ(reduce_ptr,n_reduce.data_ptr());
    
    
    // set to something in-compat
    n_reduce.set(3.1415);
    
    n_reduce.print();
    //n_reduce.reset();

    Schema s_reduce_pre;
    
    s_reduce_pre =  n_reduce.schema();
    
    mpi::all_reduce(n_snd, n_reduce, MPI_SUM, MPI_COMM_WORLD);
    
    
    n_reduce.print();
    
    reduce_vals = n_reduce.value();
    
    EXPECT_EQ(reduce_vals[0], 10 * com_size);
    EXPECT_EQ(reduce_vals[1], 20 * com_size);


    EXPECT_FALSE(s_reduce_pre.compatible(n_reduce.schema()));

}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, reduce_helpers) 
{
    int rank     = mpi::rank(MPI_COMM_WORLD);
    int com_size = mpi::size(MPI_COMM_WORLD);

    Node snd(DataType::int64(5));
    Node rcv(DataType::int64(5));
        
    int64 *snd_vals = snd.value();
    int64 *rcv_vals = rcv.value();
    
    // sum

    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = 10;
    }



    mpi::sum_reduce(snd, rcv, 0, MPI_COMM_WORLD);


    if(rank == 0)
    {
        for(int i=0; i < 5; i++)
        {
            EXPECT_EQ(rcv_vals[i], 10 * com_size);
        }
    }
    
    // prod

    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = 2;
    }


    mpi::prod_reduce(snd, rcv, 0, MPI_COMM_WORLD);


    if(rank == 0)
    {
        for(int i=0; i < 5; i++)
        {
            EXPECT_EQ(rcv_vals[i], pow(com_size,2) );
        }
    }

    
    // max
    
    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = rank * 10 + 1;
    }

    mpi::max_reduce(snd, rcv, 0, MPI_COMM_WORLD);


    if(rank == 0)
    {
        for(int i=0; i < 5; i++)
        {
            EXPECT_EQ(rcv_vals[i], 10 * (com_size-1) + 1);
        }
    }

    // min

    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = rank * 10 + 1;
    }

    mpi::min_reduce(snd, rcv, 0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        for(int i=0; i < 5; i++)
        {
            EXPECT_EQ(rcv_vals[i], 1);
        }
    }

}


//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, all_reduce_helpers) 
{
    int rank     = mpi::rank(MPI_COMM_WORLD);
    int com_size = mpi::size(MPI_COMM_WORLD);

    Node snd(DataType::int64(5));
    Node rcv(DataType::int64(5));
    
    int64 *snd_vals = snd.value();
    int64 *rcv_vals = rcv.value();

    // sum

    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = 10;
    }

    mpi::sum_all_reduce(snd, rcv, MPI_COMM_WORLD);


    for(int i=0; i < 5; i++)
    {
        EXPECT_EQ(rcv_vals[i], 10 * com_size);
    }

    // prod

    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = 2;
    }

    mpi::prod_all_reduce(snd, rcv, MPI_COMM_WORLD);


    for(int i=0; i < 5; i++)
    {
        EXPECT_EQ(rcv_vals[i], pow(com_size,2) );
    }

    
    // max
    
    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = rank * 10 + 1;
    }
    

    mpi::max_all_reduce(snd, rcv, MPI_COMM_WORLD);


    for(int i=0; i < 5; i++)
    {
        EXPECT_EQ(rcv_vals[i], 10 * (com_size-1) + 1);
    }

    // min

    for(int i=0; i < 5; i++)
    {
        snd_vals[i] = rank * 10 + 1;
    }
    

    mpi::min_all_reduce(snd, rcv, MPI_COMM_WORLD);


    for(int i=0; i < 5; i++)
    {
        EXPECT_EQ(rcv_vals[i], 1);
    }

}




//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, isend_irecv_wait) 
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
    if (rank == 0) 
    {
        mpi::irecv(n1, 1, 0, MPI_COMM_WORLD, &request);
        mpi::wait_recv(&request, &status);
    } else if (rank == 1) 
    {
        mpi::isend(n1, 0, 0, MPI_COMM_WORLD, &request);
        mpi::wait_send(&request, &status);
    }

    EXPECT_EQ(n1.as_float64_ptr()[0], 2);
    EXPECT_EQ(n1.as_float64_ptr()[1], 3.4124);
    EXPECT_EQ(n1.as_float64_ptr()[2], 9.7);
    
}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, waitall) 
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
    if (rank == 0) 
    {
        mpi::irecv(n1, 1, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::wait_all_recv(1, requests, statuses);
    } else if (rank == 1) 
    {
        mpi::isend(n1, 0, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::wait_all_send(1, requests, statuses);
    }

    EXPECT_EQ(n1.as_float64_ptr()[0], 2);
    EXPECT_EQ(n1.as_float64_ptr()[1], 3.4124);
    EXPECT_EQ(n1.as_float64_ptr()[2], 9.7);
    
}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, waitallmultirequest) 
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
    if (rank == 0) 
    {
        mpi::irecv(n1, 1, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::irecv(n2, 1, 0, MPI_COMM_WORLD, &requests[1]);
        mpi::wait_all_recv(1, requests, statuses);
    } else if (rank == 1) 
    {
        mpi::isend(n1, 0, 0, MPI_COMM_WORLD, &requests[0]);
        mpi::isend(n2, 0, 0, MPI_COMM_WORLD, &requests[1]);
        mpi::wait_all_send(1, requests, statuses);
    }

    EXPECT_EQ(n1.as_float64_ptr()[0], 2);
    EXPECT_EQ(n1.as_float64_ptr()[1], 3.4124);
    EXPECT_EQ(n1.as_float64_ptr()[2], 9.7);

    EXPECT_EQ(n2.as_int32(), 13123);
    
}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, external) 
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
    if (rank == 0) 
    {
        mpi::irecv(n1, 1, 0, MPI_COMM_WORLD, &request);
        mpi::wait_recv(&request, &status);
        
    } else if (rank == 1)
    {
        mpi::isend(n1, 0, 0, MPI_COMM_WORLD, &request);
        mpi::wait_send(&request, &status);
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
TEST(conduit_mpi_test, allgather_simple) 
{
    Node n;
    
    int rank = mpi::rank(MPI_COMM_WORLD);

    n["values/a"] = rank+1;
    n["values/b"] = rank+2;
    n["values/c"] = rank+3;
    
    Node rcv;
    mpi::all_gather(n,rcv,MPI_COMM_WORLD);
    rcv.print();
    
    Node res;
    res.set_external((int*)rcv.data_ptr(), 6);
    res.print();
    
    int *res_ptr = res.value();
    EXPECT_EQ(res_ptr[0],1);
    EXPECT_EQ(res_ptr[1],2);
    EXPECT_EQ(res_ptr[2],3);
    EXPECT_EQ(res_ptr[3],2);
    EXPECT_EQ(res_ptr[4],3);
    EXPECT_EQ(res_ptr[5],4);

}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, gather_simple) 
{
    Node n;
    
    int rank = mpi::rank(MPI_COMM_WORLD);

    n["values/a"] = rank+1;
    n["values/b"] = rank+2;
    n["values/c"] = rank+3;
    
    Node rcv;
    mpi::all_gather(n,rcv,MPI_COMM_WORLD);
    rcv.print();
    
    if(rank == 0)
    {
        Node res;
        res.set_external((int*)rcv.data_ptr(), 6);
        res.print();
    
        int *res_ptr = res.value();
        EXPECT_EQ(res_ptr[0],1);
        EXPECT_EQ(res_ptr[1],2);
        EXPECT_EQ(res_ptr[2],3);
        EXPECT_EQ(res_ptr[3],2);
        EXPECT_EQ(res_ptr[4],3);
        EXPECT_EQ(res_ptr[5],4);
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, gatherv_simple) 
{
    Node n;
    
    int rank = mpi::rank(MPI_COMM_WORLD);

    n["values/a"] = rank+1;
    n["values/b"] = rank+2;
    n["values/c"] = rank+3;
    if(rank != 0)
    {
        n["values/d"] = rank+4;
    }
    
    Node rcv;
    mpi::gatherv(n,rcv,0,MPI_COMM_WORLD);
    rcv.print();
    
    if( rank == 0)
    {
        Node res;
        res.set_external((int*)rcv.data_ptr(), 7);
        res.print();
        
        int *res_ptr = res.value();
        EXPECT_EQ(res_ptr[0],1);
        EXPECT_EQ(res_ptr[1],2);
        EXPECT_EQ(res_ptr[2],3);
        EXPECT_EQ(res_ptr[3],2);
        EXPECT_EQ(res_ptr[4],3);
        EXPECT_EQ(res_ptr[5],4);
        EXPECT_EQ(res_ptr[6],5);
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, allgatherv_simple) 
{
    Node n;
    
    int rank = mpi::rank(MPI_COMM_WORLD);

    n["values/a"] = rank+1;
    n["values/b"] = rank+2;
    n["values/c"] = rank+3;
    if(rank != 0)
    {
        n["values/d"] = rank+4;
    }
    
    Node rcv;
    mpi::all_gatherv(n,rcv,MPI_COMM_WORLD);
    rcv.print();
    
    Node res;
    res.set_external((int*)rcv.data_ptr(), 7);
    res.print();
    
    int *res_ptr = res.value();
    EXPECT_EQ(res_ptr[0],1);
    EXPECT_EQ(res_ptr[1],2);
    EXPECT_EQ(res_ptr[2],3);
    EXPECT_EQ(res_ptr[3],2);
    EXPECT_EQ(res_ptr[4],3);
    EXPECT_EQ(res_ptr[5],4);
    EXPECT_EQ(res_ptr[6],5);

}


//-----------------------------------------------------------------------------
TEST(conduit_mpi_test, bcast) 
{
    int rank = mpi::rank(MPI_COMM_WORLD);
    int com_size = mpi::size(MPI_COMM_WORLD);
    
    for(int root = 0; root < com_size; root++)
    {
        Node n;

        std::vector<int64> vals;
        if(rank == root)
        {
            vals.push_back(11);
            vals.push_back(22);
            vals.push_back(33);
        
            n.set_external(vals);
        }

        mpi::broadcast(n,root,MPI_COMM_WORLD);

        int64 *vals_ptr = n.value();

        EXPECT_EQ(vals_ptr[0], 11);
        EXPECT_EQ(vals_ptr[1], 22);
        EXPECT_EQ(vals_ptr[2], 33);
    
        CONDUIT_INFO("Bcast from root = " 
                     << root  << "\n"
                     << "rank: " << rank << " res = "
                     << n.to_json());
    }


    for(int root = 0; root < com_size; root++)
    {
        Node n;

        if(rank == root)
        {
            n["a/b/c/d/e/f"] = "g";
        }

        mpi::broadcast(n,root,MPI_COMM_WORLD);

        std::string val = n["a/b/c/d/e/f"].as_string();

        EXPECT_EQ(val, "g");
    
        CONDUIT_INFO("Bcast from root = " 
                     << root  << "\n"
                     << "rank: " << rank << " res = "
                     << n.to_json());
    }
    
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

