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
/// file: t_blueprint_mesh_examples.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <math.h>
#include <iostream>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_relay, spiral_multi_file)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";
    if(!hdf5_enabled)
    {
        CONDUIT_INFO("HDF5 disabled, skipping spiral_multi_file test");
        return;
    }
    //
    // Create an example mesh.
    //
    Node data, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,data);

    // lets try with -1 to 8 files.

    // nfiles less than 1 should trigger default case
    // (n output files = n domains)
    std::ostringstream oss;
    for(int nfiles=-1; nfiles < 9; nfiles++)
    {
        CONDUIT_INFO("test nfiles = " << nfiles);
        oss.str("");
        oss << "tout_relay_sprial_mesh_save_nfiles_" << nfiles;
        std::string output_base = oss.str();
        std::string output_dir  = output_base + ".cycle_000000";
        std::string output_root = output_base + ".cycle_000000.root";

        // remove existing directory
        remove_path_if_exists(output_dir);
        remove_path_if_exists(output_root);

        Node opts;
        opts["number_of_files"] = nfiles;
        relay::io::blueprint::save_mesh(data, output_base, "hdf5", opts);

        // count the files
        //  file_%06llu.{protocol}:/domain_%06llu/...
        int nfiles_to_check = nfiles;
        if(nfiles <=0 || nfiles == 8) // expect 7 files (one per domain)
        {
            nfiles_to_check = 7;
        }

        EXPECT_TRUE(is_directory(output_dir));
        EXPECT_TRUE(is_file(output_root));

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
            oss << join_file_path(output_base + ".cycle_000000",
                                  fprefix)
                << fmt_buff << ".hdf5";
            std::string fcheck = oss.str();
            std::cout << " checking: " << fcheck << std::endl;
            EXPECT_TRUE(is_file(fcheck));
        }
    }
}



//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_relay, save_load_mesh)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";
    if(!hdf5_enabled)
    {
        CONDUIT_INFO("HDF5 disabled, skipping spiral_multi_file test");
        return;
    }

    std::string output_base = "tout_relay_mesh_save_load";
    // spiral with 3 domains
    Node data;
    conduit::blueprint::mesh::examples::spiral(3,data);

    // spiral doesn't have domain ids, lets add some so we diff clean
    data.child(0)["state/domain_id"] = 0;
    data.child(1)["state/domain_id"] = 1;
    data.child(2)["state/domain_id"] = 2;

    Node opts;
    opts["number_of_files"] = -1;
    relay::io::blueprint::save_mesh(data, output_base, "hdf5", opts);

    data.print();
    Node n_read, info;
    relay::io::blueprint::load_mesh(output_base + ".cycle_000000.root",
                                    n_read);

    n_read.print();
    // reading back in will add domain_zzzzzz names, check children of read

    data.child(0).diff(n_read.child(0),info);
    data.child(1).diff(n_read.child(1),info);
    data.child(2).diff(n_read.child(2),info);
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_relay, save_load_mesh_opts)
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    bool hdf5_enabled = io_protos["io/protocols/hdf5"].as_string() == "enabled";
    if(!hdf5_enabled)
    {
        CONDUIT_INFO("HDF5 disabled, skipping save_load_mesh_opts test");
        return;
    }

    Node data;
    blueprint::mesh::examples::braid("uniform",
                                     2,
                                     2,
                                     2,
                                     data);

    //
    // suffix
    //

    // suffix: default, cycle, none, garbage

    std::string tout_base = "tout_relay_bp_mesh_opts_suffix";

    Node opts;
    opts["file_style"] = "root_only";

    //
    opts["suffix"] = "default";

    remove_path_if_exists(tout_base + ".cycle_000100.root");
    relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts);
    EXPECT_TRUE(is_file(tout_base + ".cycle_000100.root"));


    // remove cycle from braid, default behavior will be diff
    data.remove("state/cycle");

    remove_path_if_exists(tout_base + ".root");
    relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts);
    EXPECT_TRUE(is_file( tout_base + ".root"));

    //
    opts["suffix"] = "cycle";

    remove_path_if_exists(tout_base + ".cycle_000000.root");
    relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts);
    EXPECT_TRUE(is_file( tout_base + ".cycle_000000.root"));

    //
    opts["suffix"] = "none";

    remove_path_if_exists(tout_base + ".root");
    relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts);
    EXPECT_TRUE(is_file( tout_base + ".root"));

    // this should error
    opts["suffix"] = "garbage";
    EXPECT_THROW(relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts),Error);


    //
    // file style
    //
    // default, root_only, multi_file, garbage

    tout_base = "tout_relay_bp_mesh_opts_file_style";

    opts["file_style"] = "default";
    opts["suffix"] = "none";

    remove_path_if_exists(tout_base + ".root");
    relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts);
    EXPECT_TRUE(is_file( tout_base + ".root"));

    opts["file_style"] = "root_only";

    remove_path_if_exists(tout_base + ".root");
    relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts);
    EXPECT_TRUE(is_file( tout_base + ".root"));


    opts["file_style"] = "multi_file";

    remove_path_if_exists(tout_base + ".root");
    remove_path_if_exists(tout_base);
    relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts);
    EXPECT_TRUE(is_file( tout_base + ".root"));
    EXPECT_TRUE(is_directory(tout_base));
    EXPECT_TRUE(is_file(join_file_path(tout_base,
                                       "domain_000000.hdf5")));

    opts["file_style"] = "garbage";

    EXPECT_THROW(relay::io::blueprint::save_mesh(data, tout_base, "hdf5", opts),Error);

}


