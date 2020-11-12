// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_relay.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
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

    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    std::cout<<"Rank "<<par_rank<<" of "<<par_size<<"\n";
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

    dset["coordsets/coords/origin/x"] = -10.0 + 20.0 * par_rank;
    dset["state/domain_id"] = par_rank;
    // set cycle to 0, so we can construct the correct root file
    dset["state/cycle"] = 0;

    string protocol = "hdf5";
    //string protocol = "conduit_bin";
    string output_base = "test_blueprint_mpi_relay";
    // what we want:
    // relay::mpi::io::blueprint::write_mesh(dset, output_path,"hdf5");
    conduit::relay::mpi::io::blueprint::write_mesh(dset,
                                                  output_base,
                                                  "hdf5",
                                                  comm);

    // read this back using read_mesh, should diff clean
    string output_root = output_base + ".cycle_000000.root";
    Node n_read, n_diff_info;
    conduit::relay::mpi::io::blueprint::read_mesh(output_root,
                                                  n_read,
                                                  comm);
    // diff == false, no diff == diff clean
    EXPECT_FALSE(dset.diff(n_read.child(0),n_diff_info));
}


//-----------------------------------------------------------------------------
TEST(blueprint_mpi_relay, mpi_mesh_examples_braid)
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

    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    Node dset, v_info;
    blueprint::mpi::mesh::examples::braid_uniform_multi_domain(dset,
                                                               comm);

    // check verify
    EXPECT_TRUE(blueprint::mpi::mesh::verify(dset,v_info,comm));

    // locally, expect 1 domain
    EXPECT_EQ(blueprint::mesh::number_of_domains(dset),1);
    // globally, expect par_size domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(dset,comm),par_size);

    string protocol = "hdf5";
    string output_base = "tout_blueprint_mpi_relay_braid_uniform_multi_dom";
    // what we want:
    // relay::mpi::io::blueprint::write_mesh(dset, output_path,"hdf5");
    conduit::relay::mpi::io::blueprint::write_mesh(dset,
                                                  output_base,
                                                  "hdf5",
                                                  comm);

    // read this back using read_mesh, should diff clean
    string output_root = output_base + ".cycle_000000.root";
    Node n_read, n_diff_info;
    conduit::relay::mpi::io::blueprint::read_mesh(output_root,
                                                  n_read,
                                                  comm);
    // diff == false, no diff == diff clean
    EXPECT_FALSE(dset.diff(n_read.child(0),n_diff_info));
}

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_relay, mpi_mesh_examples_spiral_5doms)
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

    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    Node dset, v_info;
    blueprint::mpi::mesh::examples::spiral_round_robin(5,
                                                       dset,
                                                       comm);

    // check verify
    EXPECT_TRUE(blueprint::mpi::mesh::verify(dset,v_info,comm));

    // locally, expect:
    //  rank 0: 3 domain
    //  rank 1: 2 domains
    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(dset),3);
    }
    else
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(dset),2);
    }

    // globally, expect 5 domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(dset,comm),5);

    string protocol = "hdf5";
    string output_base = "tout_blueprint_mpi_relay_spiral_mpi_dist_5doms";

    // make sure the files don't exist
    if(par_rank == 0)
    {
        string output_dir  = output_base + ".cycle_000000";
        string output_root = output_base + ".cycle_000000.root";

        // remove existing output
        remove_path_if_exists(output_dir);
        remove_path_if_exists(output_root);
    }
    MPI_Barrier(comm);


    // what we want:
    // relay::mpi::io::blueprint::write_mesh(dset, output_path,"hdf5");
    conduit::relay::mpi::io::blueprint::write_mesh(dset,
                                                  output_base,
                                                  "hdf5",
                                                  comm);

    // read this back using read_mesh, should diff clean
    string output_root = output_base + ".cycle_000000.root";
    Node n_read, n_diff_info;
    conduit::relay::mpi::io::blueprint::read_mesh(output_root,
                                                  n_read,
                                                  comm);

    // globally, expect 5 domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(n_read,comm),5);

    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(n_read),3);
    }
    else
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(n_read),2);
    }

    EXPECT_EQ(dset.number_of_children(),n_read.number_of_children());
    for(conduit::index_t i=0;i < dset.number_of_children();i++)
    {
        // diff == false, no diff == diff clean
        EXPECT_FALSE(dset.child(i).diff(n_read.child(i),n_diff_info));
    }

}

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_relay, mpi_mesh_examples_spiral_1dom)
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

    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    Node dset, v_info;
    blueprint::mpi::mesh::examples::spiral_round_robin(1,
                                                       dset,
                                                       comm);

    // check verify
    EXPECT_TRUE(blueprint::mpi::mesh::verify(dset,v_info,comm));

    // locally, expect:
    //  rank 0: 1 domain
    //  rank 1: 0 domains
    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(dset),1);
    }
    else
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(dset),0);
    }

    // globally, expect par_size domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(dset,comm),1);

    string protocol = "hdf5";
    string output_base = "tout_blueprint_mpi_relay_spiral_mpi_dist_1dom";
    Node opts;
    opts["file_style"] = "multi_file";

    // make sure the files don't exist
    if(par_rank == 0)
    {
        string output_dir  = output_base + ".cycle_000000";
        string output_root = output_base + ".cycle_000000.root";

        // remove existing output
        remove_path_if_exists(output_dir);
        remove_path_if_exists(output_root);
    }
    MPI_Barrier(comm);

    conduit::relay::mpi::io::blueprint::write_mesh(dset,
                                                  output_base,
                                                  "hdf5",
                                                  opts,
                                                  comm);

    // read this back using read_mesh, should diff clean
    string output_root = output_base + ".cycle_000000.root";
    Node n_read, n_diff_info;
    conduit::relay::mpi::io::blueprint::read_mesh(output_root,
                                                  n_read,
                                                  comm);

    // globally, expect 1 domain
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(n_read,comm),1);

    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(n_read),1);
    }
    else
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(n_read),0);
    }

    EXPECT_EQ(dset.number_of_children(),n_read.number_of_children());
    for(conduit::index_t i=0;i < dset.number_of_children();i++)
    {
        dset.print();
        n_read.print();
        // diff == false, no diff == diff clean
        EXPECT_FALSE(dset.child(i).diff(n_read.child(i),n_diff_info));
    }

    // globally, expect par_size domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(n_read,comm),1);
    

}


//-----------------------------------------------------------------------------
TEST(blueprint_mpi_relay, spiral_multi_file)
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
        oss << "tout_relay_mpi_spiral_mesh_nfiles_" << nfiles;

        string output_base = oss.str();

        string output_dir  = output_base + ".cycle_000000";
        string output_root = output_base + ".cycle_000000.root";

        if(par_rank == 0)
        {
            // remove existing directory
            remove_path_if_exists(output_dir);
            remove_path_if_exists(output_root);
        }

        MPI_Barrier(comm);

        Node opts;
        opts["number_of_files"] = nfiles;
        conduit::relay::mpi::io::blueprint::write_mesh(data,
                                                      output_base,
                                                      "hdf5",
                                                      opts,
                                                      comm);

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
    }

    // read this back using read_mesh
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

