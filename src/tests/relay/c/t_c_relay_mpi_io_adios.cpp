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
/// file: t_c_relay_mpi_io_adios.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_mpi_io.h"
#include "conduit_error.hpp"
#include "conduit_cpp_to_c.hpp"
#include "gtest/gtest.h"

using namespace conduit;

// Include some utility functions
#include "conduit.hpp"
#include "../adios_test_utils.hpp"

#include <mpi.h>

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_c, test_mpi_io_c_save_and_load)
{
    int ii, rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const float64 pi = 3.141592653589793;
    conduit_int8    a[] = {1,2,3,4,5};
    conduit_int16   b[] = {2,3,4,5,6};
    conduit_int32   c[] = {3,4,5,6,7};
    conduit_int64   d[] = {4,5,6,7,8};
    conduit_float32 e[] = {1.23456, 2.3456, 3.4567, 4.5678, 5.6789};
    conduit_float64 f[] = {pi, 2.*pi, 3.*pi, 4*pi, 5*pi};
    conduit_uint8   g[] = {5,6,7,8,9};
    conduit_uint16  h[] = {6,7,8,9,10};
    conduit_uint32  i[] = {7,8,9,10,11};
    conduit_uint64  j[] = {8,9,10,11,12};
    const char *k = "ADIOS";
    const char *path = "test_mpi_io_c_save_and_load.bp";
    conduit_node *out, *in, *info;

    for(ii = 0; ii < 5; ++ii)
    {
        a[ii] += (conduit_int8)rank;
        b[ii] += (conduit_int16)rank;
        c[ii] += (conduit_int32)rank;
        d[ii] += (conduit_int64)rank;
        e[ii] += (conduit_float32)rank;
        f[ii] += (conduit_float64)rank;
        g[ii] += (conduit_uint8)rank;
        h[ii] += (conduit_uint16)rank;
        i[ii] += (conduit_uint32)rank;
        j[ii] += (conduit_uint64)rank;
    }


    /* Use the C API to make a node and save to ADIOS. */
    out = conduit_node_create();
    conduit_node_set_int8_ptr(conduit_node_fetch(out, "a"), a,
        sizeof(a) / sizeof(conduit_int8));
    conduit_node_set_int16_ptr(conduit_node_fetch(out, "b"), b,
        sizeof(b) / sizeof(conduit_int16));
    conduit_node_set_int32_ptr(conduit_node_fetch(out, "c"), c,
        sizeof(c) / sizeof(conduit_int32));
    conduit_node_set_int64_ptr(conduit_node_fetch(out, "d"), d,
        sizeof(d) / sizeof(conduit_int64));
    conduit_node_set_float32_ptr(conduit_node_fetch(out, "e"), e,
        sizeof(e) / sizeof(conduit_float32));
    conduit_node_set_float64_ptr(conduit_node_fetch(out, "f"), f,
        sizeof(f) / sizeof(conduit_float64));
    conduit_node_set_uint8_ptr(conduit_node_fetch(out, "g"), g,
        sizeof(g) / sizeof(conduit_uint8));
    conduit_node_set_uint16_ptr(conduit_node_fetch(out, "h"), h,
        sizeof(h) / sizeof(conduit_uint16));
    conduit_node_set_uint32_ptr(conduit_node_fetch(out, "i"), i,
        sizeof(i) / sizeof(conduit_uint32));
    conduit_node_set_uint64_ptr(conduit_node_fetch(out, "j"), j,
        sizeof(j) / sizeof(conduit_uint64));
    conduit_node_set_char8_str(conduit_node_fetch(out, "k"), k);

    /* Save the node in parallel */
    /*if(rank == 0) conduit_node_print(out);*/
    conduit_relay_mpi_io_save(out, path, NULL, NULL, MPI_Comm_c2f(MPI_COMM_WORLD));

    /* Read the data back in. */
    in = conduit_node_create();
    conduit_relay_mpi_io_load(path, NULL, NULL, in, MPI_Comm_c2f(MPI_COMM_WORLD));
    /*if(rank == 0) conduit_node_print(in);*/
    
    info = conduit_node_create();
    EXPECT_EQ(conduit_node_diff(out, in, info, 0.0), 0);

    /* Cleanup */
    conduit_node_destroy(out);
    conduit_node_destroy(in);
    conduit_node_destroy(info);
    
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_c, test_mpi_io_c_time_series)
{
    int rank, size;
    const char *path = "test_mpi_io_c_time_series.bp";
    const char *protocol = "adios";
    int i, ts, nts = 5;
    conduit_node **out = (conduit_node **)malloc(nts * sizeof(conduit_node *));

    // Write multiple time steps to the same file.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for(ts = 0; ts < nts; ++ts)
    {
        int idx = ts*100 + rank*10;
        out[ts] = conduit_node_create();
        conduit_node_set_int(conduit_node_fetch(out[ts], "a"), idx + 1);
        conduit_node_set_int(conduit_node_fetch(out[ts], "b"), idx + 2);
        conduit_node_set_int(conduit_node_fetch(out[ts], "c/d"), idx + 3);
        conduit_node_set_int(conduit_node_fetch(out[ts], "c/e"), idx + 4);
        conduit_node_set_float(conduit_node_fetch(out[ts], "f"), 3.14159f * (float)ts);

        if(ts == 0)
            conduit_relay_mpi_io_save(out[ts], path, NULL, NULL, MPI_Comm_c2f(MPI_COMM_WORLD));
        else
            conduit_relay_mpi_io_add_step(out[ts], path,NULL, NULL, MPI_Comm_c2f(MPI_COMM_WORLD));

        // Make sure the file has the new  step.
        int qnts = conduit_relay_mpi_io_query_number_of_steps(path, 
                       MPI_Comm_c2f(MPI_COMM_WORLD));
        EXPECT_EQ(qnts, ts+1);
    }

    conduit_node *info = conduit_node_create();
    // Let each rank read back its  steps.
    for(int ts = 0; ts < nts; ++ts)
    {
        conduit_node *in = conduit_node_create();
        conduit_relay_mpi_io_load_step_and_domain(path, protocol, ts, rank, NULL, in, MPI_Comm_c2f(MPI_COMM_WORLD));

        EXPECT_EQ(conduit_node_diff(in, out[ts],info,0.0), 0);
        conduit_node_destroy(in);
    }
    conduit_node_destroy(info);

    for(i = 0; i < nts; ++i)
        conduit_node_destroy(out[i]);
    free(out);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    conduit_relay_mpi_io_initialize(MPI_Comm_c2f(MPI_COMM_WORLD));
    result = RUN_ALL_TESTS();
    conduit_relay_mpi_io_finalize(MPI_Comm_c2f(MPI_COMM_WORLD));
    MPI_Finalize();
    return result;
}
