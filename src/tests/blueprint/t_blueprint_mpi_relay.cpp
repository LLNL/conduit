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
/// file: t_blueprint_mpi_relay.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_relay_mpi_io.hpp"
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_utils.hpp"

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::relay;
using namespace conduit::relay::mpi;
using namespace conduit::utils;

using namespace std;

// //-----------------------------------------------------------------------------
// void mesh_blueprint_save(const Node &data,
//                          const std::string &path,
//                          const std::string &file_protocol)
// {
//     Node io_protos;
//     relay::io::about(io_protos["io"]);
//     bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";
//
//     // only run this test if hdf5 is enabled
//     if(!hdf5_enabled)
//     {
//         CONDUIT_INFO("hdf5 is disabled, skipping hdf5 dependent test");
//         return;
//     }
//
//     // For simplicity, this code assumes that all ranks have
//     // data and that data is NOT multi-domain.
//
//     int par_rank = mpi::rank(MPI_COMM_WORLD);
//     int par_size = mpi::size(MPI_COMM_WORLD);
//
//     // setup the directory
//     char fmt_buff[64] = {0};
//     int cycle = data["state/cycle"].to_int32();
//     snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);
//
//     std::string output_base_path = path;
//
//     ostringstream oss;
//     oss << output_base_path << ".cycle_" << fmt_buff;
//     string output_dir  =  oss.str();
//
//     bool dir_ok = false;
//
//     // let rank zero handle dir creation
//     if(par_rank == 0)
//     {
//         // check of the dir exists
//         dir_ok = is_directory(output_dir);
//
//         if(!dir_ok)
//         {
//             // if not try to let rank zero create it
//             dir_ok = create_directory(output_dir);
//         }
//     }
//
//     int local_domains, global_domains;
//     local_domains = 1;
//
//     // use an mpi sum to check if the dir exists
//     Node n_src, n_reduce;
//
//     if(dir_ok)
//         n_src = (int)1;
//     else
//         n_src = (int)0;
//
//     mpi::sum_all_reduce(n_src,
//                         n_reduce,
//                         MPI_COMM_WORLD);
//
//     dir_ok = (n_reduce.as_int() == 1);
//
//     // find out how many domains there are
//     n_src = local_domains;
//
//     mpi::sum_all_reduce(n_src,
//                         n_reduce,
//                         MPI_COMM_WORLD);
//
//     global_domains = n_reduce.as_int();
//
//     if(!dir_ok)
//     {
//         CONDUIT_ERROR("Error: failed to create directory " << output_dir);
//     }
//
//     // write out our local domain
//     uint64 domain = data["state/domain_id"].to_uint64();
//
//     snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain);
//     oss.str("");
//     oss << "domain_" << fmt_buff << "." << file_protocol;
//     string output_file  = join_file_path(output_dir,oss.str());
//     relay::io::save(data, output_file);
//
//     int root_file_writer = 0;
//
//     // let rank zero write out the root file
//     if(par_rank == root_file_writer)
//     {
//         snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);
//
//         oss.str("");
//         oss << path
//             << ".cycle_"
//             << fmt_buff
//             << ".root";
//
//         string root_file = oss.str();
//
//         string output_dir_base, output_dir_path;
//
//         rsplit_string(output_dir,
//                       "/",
//                       output_dir_base,
//                       output_dir_path);
//
//         string output_file_pattern = join_file_path(output_dir_base, "domain_%06d." + file_protocol);
//
//
//         Node root;
//         Node &bp_idx = root["blueprint_index"];
//
//         blueprint::mesh::generate_index(data,
//                                         "",
//                                         global_domains,
//                                         bp_idx["mesh"]);
//
//         // work around conduit and manually add state fields
//         // sometimes they were not present and bad things happened
//         if(data.has_path("state/cycle"))
//         {
//           bp_idx["mesh/state/cycle"] = data["state/cycle"].to_int32();
//         }
//
//         if(data.has_path("state/time"))
//         {
//           bp_idx["mesh/state/time"] = data["state/time"].to_double();
//         }
//
//         root["protocol/name"]    =  file_protocol;
//         root["protocol/version"] = "0.4.0";
//
//         root["number_of_files"]  = global_domains;
//         // for now we will save one file per domain, so trees == files
//         root["number_of_trees"]  = global_domains;
//         // TODO: make sure this is relative
//         root["file_pattern"]     = output_file_pattern;
//         root["tree_pattern"]     = "/";
//
//         relay::io::save(root,root_file,file_protocol);
//     }
// }

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_relay, basic_use)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";

    // only run this test if hdf5 is enabled
    if(!hdf5_enabled)
    {
        CONDUIT_INFO("hdf5 is disabled, skipping hdf5 dependent test");
        return;
    }

    int rank = mpi::rank(MPI_COMM_WORLD);
    int com_size = mpi::size(MPI_COMM_WORLD);
    std::cout<<"Rank "<<rank<<" of "<<com_size<<"\n";
    index_t npts_x = 10;
    index_t npts_y = 10;
    index_t npts_z = 10;

    Node dset;
    blueprint::mesh::examples::braid("uniform",
                                      npts_x,
                                      npts_y,
                                      npts_z,
                                      dset);

    // the example data set has the bounds -10 to 10 in all dims
    // Offset this along x to create mpi 'pencil'

    dset["coordsets/coords/origin/x"] = -10.0 + 20.0 * rank;
    dset["state/domain_id"] = rank;
    // set cycle to 0, so we can construct the correct root file
    dset["state/cycle"] = 0;

    string protocol = "hdf5";
    //string protocol = "conduit_bin";
    string output_base = "test_blueprint_mpi_relay";
    // what we want:
    // relay::mpi::io::blueprint::save_mesh(dset, output_path,"hdf5");
    conduit::relay::mpi::io_blueprint::save_mesh(dset,
                                                 output_base,
                                                 "hdf5",
                                                 MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    // read this back using load_mesh, should diff clean
    string output_root = output_base + ".cycle_000000.root";
    Node n_read, n_diff_info;
    conduit::relay::mpi::io_blueprint::load_mesh(output_root,
                                                 n_read,
                                                 MPI_COMM_WORLD);
    // diff == false, no diff == diff clean
    EXPECT_FALSE(dset.diff(n_read.child(0),n_diff_info));
    
    if(rank == 0)
        n_diff_info.print();

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 1)
        n_diff_info.print();
    
    MPI_Barrier(MPI_COMM_WORLD);
    
}



//-----------------------------------------------------------------------------
TEST(blueprint_mpi_relay, spiral_multi_file)
{
    //
    // Set Up MPI
    //
    int par_rank;
    int par_size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &par_rank);
    MPI_Comm_size(comm, &par_size);

    CONDUIT_INFO("Rank "
                  << par_rank
                  << " of "
                  << par_size
                  << " reporting");

    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);
    
    // rank 0 gets first 4 domains, rank 1 gets the rest
    if(par_rank == 0)
    {
        data.remove(4);
        data.remove(4);
        data.remove(4);
    }
    else if(par_rank == 1)
    {
        data.remove(0);
        data.remove(0);
        data.remove(0);
        data.remove(0);
    }
    else
    {
        // cyrus was wrong about 2 mpi ranks.
        EXPECT_TRUE(false);
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(data,verify_info));

    std::ostringstream oss;

    // lets try with -1 to 8 files.
    
    // nfiles less than 1 should trigger default case
    // (n output files = n domains)
    for(int nfiles=-1; nfiles < 9; nfiles++)
    {
        CONDUIT_INFO("[" << par_rank <<  "] test nfiles = " << nfiles);
        MPI_Barrier(comm);
        oss.str("");
        oss << "tout_relay_mpi_sprial_mesh_nfiles_" << nfiles;

        string output_base = oss.str();

        string output_dir  = output_base + ".cycle_000000";
        string output_root = output_base + ".cycle_000000.root";

        if(par_rank == 0)
        {
            // remove existing directory
            utils::remove_directory(output_dir);
            utils::remove_directory(output_root);
        }

        MPI_Barrier(comm);

        conduit::relay::mpi::io_blueprint::save_mesh(data, output_base,"hdf5",nfiles,comm);

        MPI_Barrier(comm);

        // count the files
        //  file_%06llu.{protocol}:/domain_%06llu/...
        int nfiles_to_check = nfiles;
        if(nfiles <=0 || nfiles == 8) // expect 7 files (one per domain)
        {
            nfiles_to_check = 7;
        }

        EXPECT_TRUE(conduit::utils::is_directory(output_dir));
        EXPECT_TRUE(conduit::utils::is_file(output_root));

        char fmt_buff[64] = {0};
        for(int i=0;i<nfiles_to_check;i++)
        {
            
            std::string fprefix = "file_";
            if(nfiles_to_check == 7)
            {
                // in the n domains == n files case, the file prefix is 
                // domain_
                fprefix = "domain_";
            }
            snprintf(fmt_buff, sizeof(fmt_buff), "%06d",i);
            oss.str("");
            oss << conduit::utils::join_file_path(output_base + ".cycle_000000",
                                                  fprefix)
                << fmt_buff << ".hdf5";
            std::string fcheck = oss.str();
            std::cout << " checking: " << fcheck << std::endl;
            EXPECT_TRUE(conduit::utils::is_file(fcheck));
        }
        
        MPI_Barrier(comm);
    }
    
    // read this back using load_mesh
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

