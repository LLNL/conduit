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

//-----------------------------------------------------------------------------
void
create_rectilinear_mesh_domain(Node &out, int rank)
{
    out["domain_id"] = rank;
    float64 origin[3] = {0., 0., 0.};
    float64 csize[3]   = {3., 4., 5.};
    int     dims[3]   = {4,5,6};
    // shift domains to the right.
    origin[0] = csize[0] * rank;
    // Increase domain resolution based on rank.
    dims[0] = dims[0] * (rank+1);
    dims[1] = dims[1] * (rank+1);
    dims[2] = dims[2] * (rank+1);
    add_rectilinear_mesh(out, origin, csize, dims);
}

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
    EXPECT_EQ(compare_nodes(out, in, out), true);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_mesh)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // We write a single mesh from each rank. The resulting file will have
    // multiple pieces.
    Node out;
    create_rectilinear_mesh_domain(out, rank);

    std::string path("test_mpi_mesh.bp");
    relay::mpi::io::save(out, path, MPI_COMM_WORLD);

    // Check that there are size domains in the file.
    EXPECT_EQ(relay::mpi::io::query_number_of_domains(path, MPI_COMM_WORLD), size);

    // Each MPI rank should read its local piece and that should be the same as
    // the local data that was written.
    CONDUIT_INFO("Reading domain " << rank << "/" << size << " for " << path);
    Node in;
    relay::mpi::io::load(path, in, MPI_COMM_WORLD);

    /*if(rank == 1)
    {
        out.print_detailed();
        in.print_detailed();
    }*/

    EXPECT_EQ(compare_nodes(out, in, out), true);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_read_specific_domain)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Write a domain for this rank.
    Node domain;
    create_rectilinear_mesh_domain(domain, rank);
    std::string path("test_read_specific_domain.bp"), protocol("adios");
    relay::mpi::io::save(domain, path, MPI_COMM_WORLD);

    // rdom is the domain index of the "next" rank. We'll read that domain.
    int timestep = 0;
    int rdom = (rank + 1) % size;
    Node rdomain_from_file, rdomain_we_computed;
    relay::mpi::io::load(path, protocol, timestep, rdom, 
                         rdomain_from_file, MPI_COMM_WORLD);
    create_rectilinear_mesh_domain(rdomain_we_computed, rdom);

    // Compare the node we read vs the one we computed for the
    bool compare_nodes_local = compare_nodes(rdomain_we_computed,
                                             rdomain_from_file,
                                             rdomain_we_computed);
    /*if(rank == 1)
    {
        std::cout << "rdomain_we_computed=" << rdomain_we_computed.to_json() << std::endl;
        std::cout << "rdomain_from_file=" << rdomain_from_file.to_json() << std::endl;
        std::cout << rank << ": compare_nodes_local = " << compare_nodes_local << std::endl;
    }*/
    EXPECT_EQ(compare_nodes_local, true);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_separate_ranks)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::ostringstream oss;
    oss << "test_separate_ranks_" << rank << ".bp";
    std::string path(oss.str());

    // Split the communicator into size pieces
    MPI_Comm split;
    MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &split);

    int srank, ssize;
    MPI_Comm_rank(split, &srank);
    MPI_Comm_size(split, &ssize);
    EXPECT_EQ(srank, 0);
    EXPECT_EQ(ssize, 1);

    // Use the split communicator to write/read separate files.
    Node out;
    create_rectilinear_mesh_domain(out, rank); // use global rank on purpose here
    relay::mpi::io::save(out, path, split);

    Node in;
    CONDUIT_INFO("Reading domain " << srank << "/" << ssize << " for " << path);
    relay::mpi::io::load(path, in, split);
    bool compare_nodes_local = compare_nodes(out, in, out);
    /*if(rank == 1)
    {
        std::cout << "out=" << out.to_json() << std::endl;
        std::cout << "in=" << in.to_json() << std::endl;
        std::cout << rank << ": compare_nodes_local = " << compare_nodes_local << std::endl;
    }*/
    EXPECT_EQ(compare_nodes_local, true);

    MPI_Comm_free(&split);
}
#endif

#if 0
//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_different_trees)
{
    // NOTE: Conduit+ADIOS can kind of deal with the case where the tree is
    //       a little different on each rank. The ADIOS file will contain
    //       the right data. The issue we have when not all ranks write the
    //       same data is that we can't 100% reliably assign it back to the
    //       same rank that wrote it because ADIOS stores a "process_id" of
    //       the writing process. That number is more like an index of the 
    //       rank within the communicator that had the data rather than a
    //       global rank.

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Node out;
    out["a/b"] = 5;
    out["a/pi"] = std::vector<double>(5, M_PI);
    out["a/rank"] = std::vector<int>(10, rank);
    char key[10];
    sprintf(key, "rank%d/message", rank);
    std::ostringstream oss;
    oss << "Rank " << rank << " likes ADIOS";
    for(int i = 0; i < rank+1; ++i)
        oss << " very";
    oss << " much.";
    out[key] = oss.str();
    if(rank == 0)
        out["extra"] = "stuff";

    std::string path("test_mpi_different_trees.bp");
    relay::mpi::io::save(out, path, MPI_COMM_WORLD);

    Node in;
    CONDUIT_INFO("Reading domain " << rank << "/" << size << " for " << path);
    relay::mpi::io::load(path, in, MPI_COMM_WORLD);
    bool compare_nodes_local = compare_nodes(out, in, out);
    if(rank == 1)
    {
        out.print_detailed();
        in.print_detailed();
        std::cout << "compare_nodes_local = " << compare_nodes_local << std::endl;
    }
    EXPECT_EQ(compare_nodes_local, true);
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
