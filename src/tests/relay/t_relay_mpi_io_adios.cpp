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

//-----------------------------------------------------------------------------
void
mpi_print_node(const Node &node, const std::string &name, MPI_Comm comm)
{
    static int tag = 12345;
    int rank, size, msg;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if(rank == 0)
    {
        std::cout << rank << ": " << name << " = " << node.to_json() << std::endl;
        MPI_Send(&rank, 1, MPI_INT, rank+1, tag, comm);
    }
    else
    {
        MPI_Status status;
        MPI_Recv(&msg, 1, MPI_INT, rank-1, tag, comm, &status);
        std::cout << rank << ": " << name << " = " << node.to_json() << std::endl;
        if(rank < size-1)
            MPI_Send(&rank, 1, MPI_INT, rank+1, tag, comm);
    }
    tag++;
}

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

    Node n_info;
    // make sure there is no diff (diff res == FALSE)
    EXPECT_FALSE(out.diff(in,n_info,0.0));
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

    Node n_info;
    // make sure there is no diff (diff res == FALSE)
    EXPECT_FALSE(out.diff(in,n_info,0.0));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_read_specific_domain)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Write a domain for this rank.
    Node domain;
    create_rectilinear_mesh_domain(domain, rank);
    std::string path("test_mpi_read_specific_domain.bp"), protocol("adios");
    relay::mpi::io::save(domain, path, MPI_COMM_WORLD);

    // rdom is the domain index of the "next" rank. We'll read that domain.
    int timestep = 0;
    int rdom = (rank + 1) % size;
    Node rdomain_from_file, rdomain_we_computed;
    relay::mpi::io::load(path, protocol, timestep, rdom, 
                         rdomain_from_file, MPI_COMM_WORLD);
    create_rectilinear_mesh_domain(rdomain_we_computed, rdom);


    Node n_info;
    // make sure there is no diff (diff res == FALSE)
    EXPECT_FALSE(rdomain_we_computed.diff(rdomain_from_file,n_info,0.0));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_separate_ranks)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::ostringstream oss;
    oss << "test_mpi_separate_ranks_" << rank << ".bp";
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
    
    

    Node n_info;
    // make sure there is no diff (diff res == FALSE)
    EXPECT_FALSE(out.diff(in,n_info,0.0));
    /*if(rank == 1)
    {
        std::cout << "out=" << out.to_json() << std::endl;
        std::cout << "in=" << in.to_json() << std::endl;
        std::cout << rank << ": compare_nodes_local = " << compare_nodes_local << std::endl;
    }*/

    MPI_Comm_free(&split);
}

TEST(conduit_relay_mpi_io_adios, test_mpi_load_subtree)
{
    const char *abc_keys[] = {"a/b/c/d","a/b/c/dog"};
    const char *a_keys[] = {"a/b/cat",
                            "a/b/carnivores/cat",
                            "a/b/carnivores/dinosaur"};
    const char *aa_keys[] = {"a/a/bull"};
    const char *b_keys[] = {"b/c/d/e", "binary", "blue"};

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Make a node we can save.
    int index = rank * 100;
    Node out;
    add_keys_to_node(out, abc_keys, 2, index);
    add_keys_to_node(out, a_keys, 3, index);
    add_keys_to_node(out, aa_keys, 1, index);
    add_keys_to_node(out, b_keys, 3, index);
    std::string path("test_mpi_load_subtree.bp");
    relay::mpi::io::save(out, path, MPI_COMM_WORLD);
    //mpi_print_node(out, "out", MPI_COMM_WORLD);

    // Try reading nodes with subpath a/b for the same domain
    // we wrote on this rank.
    Node in;
    std::string path_ab(path + ":a/b");
    relay::mpi::io::load(path_ab, in, MPI_COMM_WORLD);
    Node ab;
    copy_node_keys(ab, out, abc_keys, 2);
    copy_node_keys(ab, out, a_keys, 3);
    
    Node n_info;
    
    //mpi_print_node(in, "in", MPI_COMM_WORLD);
    // EXPECT_EQ(compare_nodes(ab, in, ab), true);
    EXPECT_FALSE(ab.diff(in,n_info,0.0));

    // Make what rank 0 would have written for a/b.
    int index0 = 0;
    Node out0;
    add_keys_to_node(out0, abc_keys, 2, index0);
    add_keys_to_node(out0, a_keys, 3, index0);

    // Read the data that rank 0 wrote for a/b.
    Node in0;
    std::string path_ab0(path + ":0:a/b");
    relay::mpi::io::load(path_ab0, in0, MPI_COMM_WORLD);
    //mpi_print_node(in0, "in0", MPI_COMM_WORLD);
    // EXPECT_EQ(compare_nodes(out0, in0, out0), true);
    // make sure there is no diff (diff res == FALSE)
    EXPECT_FALSE(out0.diff(in0,n_info,0.0));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_adios, test_mpi_write_serial_read_parallel)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string path("test_mpi_write_serial_read_parallel.bp");
    Node *out = new Node[size];
    for(int dom = 0; dom < size; ++dom)
        create_rectilinear_mesh_domain(out[dom], dom);

    // Split the communicator into size pieces
    MPI_Comm split;
    MPI_Comm_split(MPI_COMM_WORLD, rank, 0, &split);

    // Write the data serially on rank 0 using the split comm.
    if(rank == 0)
    {
        for(int dom = 0; dom < size; ++dom)
        {
            if(dom == 0)
                relay::mpi::io::save(out[dom], path, split);
            else
                relay::mpi::io::save_merged(out[dom], path, split);
        }
    }

    // Wait for the file to be ready.
    MPI_Barrier(MPI_COMM_WORLD);

    // Now, read the rank'th domain on this rank, using the parallel API
    Node in;
    relay::mpi::io::load(path, in, MPI_COMM_WORLD);
    //mpi_print_node(in, "in", MPI_COMM_WORLD);

    Node n_info;
    
    //mpi_print_node(in, "in", MPI_COMM_WORLD);
    EXPECT_FALSE(out[rank].diff(in,n_info,0.0));
    // EXPECT_EQ(compare_nodes(out[rank], in, out[rank]), true);
    delete [] out;
    MPI_Comm_free(&split);
}

TEST(conduit_relay_io_adios, test_mpi_time_series)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::string path("test_mpi_time_series.bp"), protocol("adios");

    // Remove the file if it exists.
    conduit::utils::remove_file(path);
    MPI_Barrier(MPI_COMM_WORLD);

    // Write multiple time steps to the same file.
    int nts = 5;
    Node *out = new Node[nts];
    for(int ts = 0; ts < nts; ++ts)
    {
        Node &n = out[ts];
        int idx = ts*100 + rank*10;
        n["a"] = idx + 1;
        n["b"] = idx + 2;
        n["c/d"] = idx + 3;
        n["c/e"] = idx + 4;
        n["f"] = 3.14159f * float(ts);

        /*std::ostringstream oss;
        oss << "ts" << ts;
        mpi_print_node(n, oss.str(), MPI_COMM_WORLD);*/
        relay::mpi::io::add_step(n, path, MPI_COMM_WORLD);

        // Make sure the file has the new time step.
        int qnts = relay::mpi::io::query_number_of_steps(path, MPI_COMM_WORLD);
        EXPECT_EQ(qnts, ts+1);
    }

    // Let each rank read back its time steps.
    for(int ts = 0; ts < nts; ++ts)
    {
        Node in;
        relay::mpi::io::load(path, protocol, ts, rank, in, MPI_COMM_WORLD);
        /*std::ostringstream oss;
        oss << "ts" << ts;
        mpi_print_node(in, oss.str(), MPI_COMM_WORLD);*/
        Node n_info;
        EXPECT_FALSE(in.diff(out[ts],n_info,0.0));
        //EXPECT_EQ(compare_nodes(in, out[ts], in), true);
    }

    delete [] out;
}

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
    out["a"] = 1;
    out["b"] = std::vector<double>(5, M_PI);
    out["c"] = std::vector<int>(10, rank);
    std::ostringstream oss;
    oss << "Rank " << rank << " likes ADIOS";
    for(int i = 0; i < rank+1; ++i)
        oss << " very";
    oss << " much.";
    out["message"] = oss.str();
    if(rank == 0)
    {
        // We can add extra stuff to rank 0's output okay.
        out["diagnostics/date"] = "today";
        out["diagnostics/history/data"] = std::vector<double>(5, 1.2345);
        out["diagnostics/history/n"] = 5;
        out["diagnostics/timers/start"] = 0;
        out["diagnostics/timers/end"] = 99;
    }
#if 0
    else
    {
        // TO FIX:
        // We cannot add extra stuff to non-rank 0's output okay and 
        // read it back to the right rank. If we add this now, the
        // node would end up in the readback data for domain 0, which
        // is not right since domain 1 contained the data.
        out["non-rank-0/rank"] = rank;
    }
#endif

    std::string path("test_mpi_different_trees.bp");
    relay::mpi::io::save(out, path, MPI_COMM_WORLD);
    //mpi_print_node(out, "out", MPI_COMM_WORLD);

    Node in;
    CONDUIT_INFO("Reading domain " << rank << "/" << size << " for " << path);
    relay::mpi::io::load(path, in, MPI_COMM_WORLD);
    
    Node n_info;
    EXPECT_FALSE(out.diff(in,n_info,0.0));
    
    // bool compare_nodes_local = compare_nodes(out, in, out);
    // //mpi_print_node(in, "in", MPI_COMM_WORLD);
    // //std::cout << "compare_nodes_local = " << compare_nodes_local << std::endl;
    // EXPECT_EQ(compare_nodes_local, true);
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int result = 0;

    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    conduit::relay::mpi::io::initialize(MPI_COMM_WORLD);
    result = RUN_ALL_TESTS();
    conduit::relay::mpi::io::finalize(MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
