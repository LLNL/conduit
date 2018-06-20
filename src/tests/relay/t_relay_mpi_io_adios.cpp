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
/// file: t_relay_io_adios.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_mpi_io.hpp"
#include <iostream>
#include <cmath>
#include "gtest/gtest.h"

#include <mpi.h>

using namespace conduit;

// Include some utility functions
#include "adios_test_utils.hpp"

#if 1
//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_rank_values)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Make a string that is a different size on each rank.
    std::ostringstream oss;
    oss << "Rank " << rank << " likes ADIOS";
    for(int i = 0; i < rank+1; ++i)
        oss << " very";
    oss << " much.";

    // Make some data that is different on each processor.
    int8    a((int8)rank);
    int16   b((int16)rank);
    int32   c((int32)rank);
    int64   d((int64)rank);
    float32 e((float32)rank);
    float64 f((float64)rank);
    uint8   g((uint8)rank);
    uint16  h((uint16)rank);
    uint32  i((uint32)rank);
    uint64  j((uint64)rank);
    std::string k(oss.str());
    int8    l[] = {0, 1, 2,50,51,52};
    int16   m[] = {0, 1, 2};
    int32   n[] = {0, 1, 2};
    int64   o[] = {0, 1, 2};
    float32 p[] = {0.f, 1.f, 2.f};
    float64 q[] = {0., 1., 2.};
    for(int ii = 0; ii < 3; ++ii)
    {
        l[ii] += (int8)rank;
        m[ii] += (int16)rank;
        n[ii] += (int32)rank;
        o[ii] += (int64)rank;
        p[ii] += (float32)rank;
        q[ii] += (float64)rank;
    }

    Node out;
    out["a"] = a;
    out["b"] = b;
    out["c"] = c;
    out["d"] = d;
    out["e"] = e;
    out["f"] = f;
    out["g"] = g;
    out["h"] = h;
    out["i"] = i;
    out["j"] = j;
    out["k"] = k;
    out["l"].set(l, (rank==0) ? 3 : 6);
    out["m"].set(m, 3);
    out["n"].set(n, 3);
    out["o"].set(o, 3);
    out["p"].set(p, 3);
    out["q"].set(q, 3);

    // Save out data from each rank to a single file. Each of the variables
    // in a node will have multiple pieces, contributions from each rank.
    std::string path("test_mpi_rank_values.bp");
    relay::mpi::io::save(out, path, MPI_COMM_WORLD);

    // Have each rank read its part of the data back in.
    Node in;
    relay::mpi::io::load(path, in, MPI_COMM_WORLD);
    /*
    if(rank == 0)
    {
        std::cout << out.to_json() << std::endl;
        std::cout << in.to_json() << std::endl;
    }
    */
    // Make sure the data that was read back in is the same as the written data.
    int compare_nodes_local  = compare_nodes(out, in, out);
    EXPECT_EQ(compare_nodes_local, 1);
}
#endif

#if 0
//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_mesh)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // We write a single mesh from each rank. The resulting file will have
    // multiple pieces.
    Node out;
    out["domain_id"] = rank;
    float64 origin[3] = {0., 0., 0.};
    float64 csize[3]   = {3., 4., 5.};
    int     dims[3]   = {4,5,6};
    // shift domains to the right.
    origin[0] = csize[0] * rank;
#if 0
    // Increase domain resolution based on rank.
    dims[0] = dims[0] * (rank+1);
    dims[1] = dims[1] * (rank+1);
    dims[2] = dims[2] * (rank+1);
#endif
    add_rectilinear_mesh(out, origin, csize, dims);

    out.print_detailed();
    std::string path("test_mpi_mesh.bp");
    relay::mpi::io::save(out, path, MPI_COMM_WORLD);
#if 0
    // Check that there are size domains in the file.
    EXPECT_EQ(relay::mpi::io::number_of_domains(path, MPI_COMM_WORLD), size);
#endif
    // Each MPI rank should read its local piece and that should be the same as
    // the local data that was written.
    CONDUIT_INFO("Reading domain " << rank << "/" << size << " for " << path);
    Node in;
/*
    Node selection;
    selection["domain_id"] = rank;
    selection["type"] = "select_by_domain_id";
*/
    relay::mpi::io::load(path, in, MPI_COMM_WORLD);
    int compare_nodes_local  = compare_nodes(out, in, out);

if(rank == 1)
{
    std::cout << out.to_json() << std::endl;
    std::cout << in.to_json() << std::endl;
}

//    int compare_nodes_global = 0;
//    MPI_Allreduce(&compare_nodes_local, &compare_nodes_global, 1, MPI_INT,
//        MPI_SUM, MPI_COMM_WORLD);
    EXPECT_EQ(compare_nodes_local, 1);
}
#endif

#if 0
//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_mesh)
{
    // Write a 2 domain mesh in parallel

    // split the comm into 2 parts.

    // Have rank 0 read rank 1's mesh.
    // Have rank 1 read rank 0's mesh.
}
#endif

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

