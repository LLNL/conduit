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

//-----------------------------------------------------------------------------
void mesh_blueprint_save(const Node &data,
                         const std::string &path,
                         const std::string &file_protocol)
{
    // For simplicity, this code assumes that all ranks have
    // data and that data is NOT multi-domain.

    int par_rank = mpi::rank(MPI_COMM_WORLD);
    int par_size = mpi::size(MPI_COMM_WORLD);

    // setup the directory
    char fmt_buff[64];
    int cycle = data["state/cycle"].to_int32();
    snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);

    std::string output_base_path = path;

    ostringstream oss;
    oss << output_base_path << ".cycle_" << fmt_buff;
    string output_dir  =  oss.str();

    bool dir_ok = false;

    // let rank zero handle dir creation
    if(par_rank == 0)
    {
        // check of the dir exists
        dir_ok = is_directory(output_dir);

        if(!dir_ok)
        {
            // if not try to let rank zero create it
            dir_ok = create_directory(output_dir);
        }
    }

    int local_domains, global_domains;
    local_domains = 1;

    // use an mpi sum to check if the dir exists
    Node n_src, n_reduce;

    if(dir_ok)
        n_src = (int)1;
    else
        n_src = (int)0;

    mpi::sum_all_reduce(n_src,
                        n_reduce,
                        MPI_COMM_WORLD);

    dir_ok = (n_reduce.as_int() == 1);

    // find out how many domains there are
    n_src = local_domains;

    mpi::sum_all_reduce(n_src,
                        n_reduce,
                        MPI_COMM_WORLD);

    global_domains = n_reduce.as_int();

    if(!dir_ok)
    {
        CONDUIT_ERROR("Error: failed to create directory " << output_dir);
    }

    // write out our local domain
    uint64 domain = data["state/domain_id"].to_uint64();

    snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain);
    oss.str("");
    oss << "domain_" << fmt_buff << "." << file_protocol;
    string output_file  = join_file_path(output_dir,oss.str());
    relay::io::save(data, output_file);

    int root_file_writer = 0;

    // let rank zero write out the root file
    if(par_rank == root_file_writer)
    {
        snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);

        oss.str("");
        oss << path
            << ".cycle_"
            << fmt_buff
            << ".root";

        string root_file = oss.str();

        string output_dir_base, output_dir_path;

        rsplit_string(output_dir,
                      "/",
                      output_dir_base,
                      output_dir_path);

        string output_file_pattern = join_file_path(output_dir_base, "domain_%06d." + file_protocol);


        Node root;
        Node &bp_idx = root["blueprint_index"];

        blueprint::mesh::generate_index(data,
                                        "",
                                        global_domains,
                                        bp_idx["mesh"]);

        // work around conduit and manually add state fields
        // sometimes they were not present and bad things happened
        if(data.has_path("state/cycle"))
        {
          bp_idx["mesh/state/cycle"] = data["state/cycle"].to_int32();
        }

        if(data.has_path("state/time"))
        {
          bp_idx["mesh/state/time"] = data["state/time"].to_double();
        }

        root["protocol/name"]    =  file_protocol;
        root["protocol/version"] = "0.4.0";

        root["number_of_files"]  = global_domains;
        // for now we will save one file per domain, so trees == files
        root["number_of_trees"]  = global_domains;
        // TODO: make sure this is relative
        root["file_pattern"]     = output_file_pattern;
        root["tree_pattern"]     = "/";

        relay::io::save(root,root_file,file_protocol);
    }
}
//-----------------------------------------------------------------------------
TEST(blueprint_mpi_relay, basic_use)
{
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

    string protocol = "hdf5";
    //string protocol = "conduit_bin";
    string output_path = "test_blueprint_mpi_relay";
    mesh_blueprint_save(dset, output_path, protocol);
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

