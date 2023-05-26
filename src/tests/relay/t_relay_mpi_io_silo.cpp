// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_relay_mpi_io_silo.cpp
///
//-----------------------------------------------------------------------------

#include "silo_test_utils.hpp"

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_relay_mpi_io.hpp"
#include "conduit_relay_mpi_io_silo.hpp"
#include "conduit_fmt/conduit_fmt.h"

#include <iostream>
#include "gtest/gtest.h"

#include <mpi.h>


using namespace conduit;
using namespace conduit::utils;
using namespace conduit::relay;

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_silo, round_trip_basic)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    std::cout<<"Rank "<<par_rank<<" of "<<par_size<<"\n";

    index_t npts_x = 10;
    index_t npts_y = 10;
    index_t npts_z = 10;

    std::string mesh_type = "uniform";

    Node save_mesh;
    blueprint::mesh::examples::braid(mesh_type,
                                      npts_x,
                                      npts_y,
                                      npts_z,
                                      save_mesh);

    // the example data set has the bounds -10 to 10 in all dims
    // Offset this along x to create mpi 'pencil'

    save_mesh["coordsets/coords/origin/x"] = -10.0 + 20.0 * par_rank;
    save_mesh["state/domain_id"] = par_rank;
    // set cycle to 0, so we can construct the correct root file
    save_mesh["state/cycle"] = (int64) 0;

    Node load_mesh, info;

    const std::string basename = "silo_mpi_braid_" + mesh_type + "_3D";
    const std::string filename = basename + ".cycle_000000.root";

    remove_path_if_exists(filename);
    relay::mpi::io::silo::save_mesh(save_mesh, basename, comm);
    relay::mpi::io::silo::load_mesh(filename, load_mesh, comm);

    EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

    silo_uniform_to_rect_conversion("coords", "mesh", save_mesh);

    // make changes to save mesh so the diff will pass
    silo_name_changer("mesh", save_mesh);

    // the loaded mesh will be in the multidomain format
    // but the saved mesh is in the single domain format
    EXPECT_EQ(load_mesh.number_of_children(), 1);
    EXPECT_EQ(load_mesh.child(0).number_of_children(), save_mesh.number_of_children());

    EXPECT_FALSE(load_mesh.child(0).diff(save_mesh, info));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_silo, mpi_mesh_examples_braid)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    Node save_mesh, v_info;
    blueprint::mpi::mesh::examples::braid_uniform_multi_domain(save_mesh,
                                                               comm);
    save_mesh["state/cycle"] = (int64) 0;

    // check verify
    EXPECT_TRUE(blueprint::mpi::mesh::verify(save_mesh,v_info,comm));

    // locally, expect 1 domain
    EXPECT_EQ(blueprint::mesh::number_of_domains(save_mesh),1);
    // globally, expect par_size domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(save_mesh,comm),par_size);

    std::string output_base = "silo_mpi_braid_uniform_multi_dom";
    conduit::relay::mpi::io::silo::save_mesh(save_mesh,
                                             output_base,
                                             comm);

    // read this back using read_mesh, should diff clean
    std::string output_root = output_base + ".cycle_000000.root";
    Node load_mesh, info;
    conduit::relay::mpi::io::silo::load_mesh(output_root,
                                                  load_mesh,
                                                  comm);

    silo_uniform_to_rect_conversion("coords", "mesh", save_mesh);

    // make changes to save mesh so the diff will pass
    silo_name_changer("mesh", save_mesh);

    // diff == false, no diff == diff clean
    EXPECT_FALSE(save_mesh.diff(load_mesh.child(0),info));
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_silo, mpi_mesh_examples_spiral_5doms)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    Node save_mesh, info;
    blueprint::mpi::mesh::examples::spiral_round_robin(5,
                                                       save_mesh,
                                                       comm);

    // check verify
    EXPECT_TRUE(blueprint::mpi::mesh::verify(save_mesh,info,comm));

    // locally, expect:
    //  rank 0: 3 domain
    //  rank 1: 2 domains
    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(save_mesh),3);
        std::cout << "[rank 0] input domain ids: " << std::endl;
        save_mesh.child(0)["state/domain_id"].print();
        save_mesh.child(1)["state/domain_id"].print();
        save_mesh.child(2)["state/domain_id"].print();
    }
    MPI_Barrier(comm);
    if(par_rank == 1)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(save_mesh),2);
        std::cout << "[rank 1] input domain ids: " << std::endl;
        save_mesh.child(0)["state/domain_id"].print();
        save_mesh.child(1)["state/domain_id"].print();
    }
    MPI_Barrier(comm);

    // globally, expect 5 domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(save_mesh,comm),5);

    std::string output_base = "silo_mpi_spiral_3doms";

    // make sure the files don't exist
    if(par_rank == 0)
    {
        std::string output_dir  = output_base + ".cycle_000000";
        std::string output_root = output_base + ".cycle_000000.root";

        // remove existing output
        remove_path_if_exists(output_dir);
        remove_path_if_exists(output_root);
    }
    MPI_Barrier(comm);

    conduit::relay::mpi::io::silo::save_mesh(save_mesh,
                                             output_base,
                                             comm);

    // read this back using read_mesh.
    // note the domain ids will change, so we don't expect
    // this to diff clean
    
    std::string output_root = output_base + ".cycle_000000.root";
    Node load_mesh;
    conduit::relay::mpi::io::silo::load_mesh(output_root,
                                             load_mesh,
                                             comm);

    // globally, expect 5 domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(load_mesh,comm),5);

    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(load_mesh),3);
        std::cout << "[rank 0] read domain ids: " << std::endl;
        load_mesh.child(0)["state/domain_id"].print();
        load_mesh.child(1)["state/domain_id"].print();
        load_mesh.child(2)["state/domain_id"].print();
        // expect we bring back domains 0 - 2
        EXPECT_EQ(load_mesh.child(0)["state/domain_id"].to_index_t(),0);
        EXPECT_EQ(load_mesh.child(1)["state/domain_id"].to_index_t(),1);
        EXPECT_EQ(load_mesh.child(2)["state/domain_id"].to_index_t(),2);
    }
    MPI_Barrier(comm);
    if(par_rank == 1)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(load_mesh),2);
        std::cout << "[rank 1] read domain ids: " << std::endl;
        load_mesh.child(0)["state/domain_id"].print();
        load_mesh.child(1)["state/domain_id"].print();
        // expect we bring back domains 3 - 4
        EXPECT_EQ(load_mesh.child(0)["state/domain_id"].to_index_t(),3);
        EXPECT_EQ(load_mesh.child(1)["state/domain_id"].to_index_t(),4);
    }
    MPI_Barrier(comm);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_silo, mpi_mesh_examples_spiral_1dom)
{
    MPI_Comm comm = MPI_COMM_WORLD;
    int par_rank = mpi::rank(comm);
    int par_size = mpi::size(comm);

    Node save_mesh, info;
    blueprint::mpi::mesh::examples::spiral_round_robin(1,
                                                       save_mesh,
                                                       comm);

    // check verify
    EXPECT_TRUE(blueprint::mpi::mesh::verify(save_mesh,info,comm));

    // locally, expect:
    //  rank 0: 1 domain
    //  rank 1: 0 domains
    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(save_mesh),1);
    }
    else
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(save_mesh),0);
    }

    // globally, expect par_size domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(save_mesh,comm),1);

    std::string output_base = "silo_mpi_spiral_1dom";
    Node opts;
    opts["file_style"] = "multi_file";

    // make sure the files don't exist
    if(par_rank == 0)
    {
        std::string output_dir  = output_base + ".cycle_000000";
        std::string output_root = output_base + ".cycle_000000.root";

        // remove existing output
        remove_path_if_exists(output_dir);
        remove_path_if_exists(output_root);
    }
    MPI_Barrier(comm);

    conduit::relay::mpi::io::silo::save_mesh(save_mesh,
                                              output_base,
                                              opts,
                                              comm);

    // make changes to save mesh so the diff will pass
    for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
    {
        silo_name_changer("mesh", save_mesh[child]);
        int cycle = save_mesh[child]["state"]["cycle"].as_int32();
        save_mesh[child]["state"]["cycle"].reset();
        save_mesh[child]["state"]["cycle"] = (int64) cycle;
    }

    // read this back using read_mesh, should diff clean
    std::string output_root = output_base + ".cycle_000000.root";
    Node load_mesh, n_diff_info;
    conduit::relay::mpi::io::silo::load_mesh(output_root,
                                             load_mesh,
                                             comm);

    // globally, expect 1 domain
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(load_mesh,comm),1);

    if(par_rank == 0)
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(load_mesh),1);
    }
    else
    {
        EXPECT_EQ(blueprint::mesh::number_of_domains(load_mesh),0);
    }

    EXPECT_EQ(save_mesh.number_of_children(),load_mesh.number_of_children());
    for(conduit::index_t i=0;i < save_mesh.number_of_children();i++)
    {
        save_mesh.print();
        load_mesh.print();
        // diff == false, no diff == diff clean
        EXPECT_FALSE(save_mesh.child(i).diff(load_mesh.child(i),n_diff_info));
    }

    // globally, expect par_size domains
    EXPECT_EQ(blueprint::mpi::mesh::number_of_domains(load_mesh,comm),1);
}

//-----------------------------------------------------------------------------
TEST(conduit_relay_mpi_io_silo, spiral_multi_file)
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
    Node save_mesh, verify_info;

    // use spiral , with 7 domains
    conduit::blueprint::mesh::examples::spiral(7,save_mesh);

    // rank 0 gets first 4 domains, rank 1 gets the rest
    if(par_rank == 0)
    {
        save_mesh.remove(4);
        save_mesh.remove(4);
        save_mesh.remove(4);
    }
    else if(par_rank == 1)
    {
        save_mesh.remove(0);
        save_mesh.remove(0);
        save_mesh.remove(0);
        save_mesh.remove(0);
    }
    else
    {
        // cyrus was wrong about 2 mpi ranks.
        EXPECT_TRUE(false);
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(save_mesh,verify_info));

    std::ostringstream oss;

    // lets try with -1 to 8 files.

    // nfiles less than 1 should trigger default case
    // (n output files = n domains)
    for(int nfiles=-1; nfiles < 9; nfiles++)
    {
        CONDUIT_INFO("[" << par_rank <<  "] test nfiles = " << nfiles);
        MPI_Barrier(comm);
        oss.str("");
        oss << "silo_mpi_spiral_nfiles_" << nfiles;

        std::string output_base = oss.str();

        std::string output_dir  = output_base + ".cycle_000000";
        std::string output_root = output_base + ".cycle_000000.root";

        int nfiles_to_check = nfiles;
        if(nfiles <=0 || nfiles == 8) // expect 7 files (one per domain)
        {
            nfiles_to_check = 7;
        }

        if(par_rank == 0)
        {
            // remove existing root, output files and directory
            remove_path_if_exists(output_root);
            for(int i=0;i<nfiles_to_check;i++)
            {

                std::string fprefix = "file_";
                if(nfiles_to_check == 7)
                {
                    // in the n domains == n files case, the file prefix is
                    // domain_
                    fprefix = "domain_";
                }

                std::string output_file = conduit_fmt::format("{}{:06d}.silo",
                                join_file_path(output_base + ".cycle_000000",
                                               fprefix),
                                i);
                remove_path_if_exists(output_file);
            }

            remove_path_if_exists(output_dir);
        }

        MPI_Barrier(comm);

        Node opts;
        opts["number_of_files"] = nfiles;
        conduit::relay::mpi::io::silo::save_mesh(save_mesh,
                                                 output_base,
                                                 opts,
                                                 comm);

        MPI_Barrier(comm);

        // count the files
        //  file_%06llu.{protocol}:/domain_%06llu/...


        EXPECT_TRUE(conduit::utils::is_directory(output_dir));
        EXPECT_TRUE(conduit::utils::is_file(output_root));

        for(int i=0;i<nfiles_to_check;i++)
        {

            std::string fprefix = "file_";
            if(nfiles_to_check == 7)
            {
                // in the n domains == n files case, the file prefix is
                // domain_
                fprefix = "domain_";
            }

            std::string fcheck = conduit_fmt::format("{}{:06d}.silo",
                            join_file_path(output_base + ".cycle_000000",
                                           fprefix),
                            i);

            std::cout << " checking: " << fcheck << std::endl;
            EXPECT_TRUE(conduit::utils::is_file(fcheck));
        }

        // read the mesh back in diff to make sure we have the same save_mesh
        Node load_mesh, info;
        relay::mpi::io::silo::load_mesh(output_base + ".cycle_000000.root",
                                             load_mesh,
                                             comm);

        // make changes to save mesh so the diff will pass
        for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
        {
            silo_name_changer("mesh", save_mesh[child]);

            if (save_mesh[child]["state"]["cycle"].dtype().is_int32())
            {
                int cycle = save_mesh[child]["state"]["cycle"].as_int32();
                save_mesh[child]["state"]["cycle"].reset();
                save_mesh[child]["state"]["cycle"] = (int64) cycle;
            }
        }

        // rank 0 will have 4, rank 1 wil have 3
        int num_local_domains = 4;
        if(par_rank != 0)
        {
            num_local_domains = 3;
        }

        // total doms should be 7
        EXPECT_EQ( conduit::blueprint::mpi::mesh::number_of_domains(load_mesh, comm), 7);

        std::cout << "par_rank " << par_rank << "  read # of children " << load_mesh.number_of_children();
        // in all cases we expect 7 domains to match
        for(int dom_idx =0; dom_idx <num_local_domains; dom_idx++)
        {
            EXPECT_FALSE(save_mesh.child(dom_idx).diff(load_mesh.child(dom_idx),info));
        }

    }

    // read this back using read_mesh
}

// //-----------------------------------------------------------------------------
// // 
// // special case tests
// // 

// //-----------------------------------------------------------------------------
// // var is not defined on a domain
// // 
// // tests the silo "EMPTY" capability
// TEST(conduit_relay_io_silo, missing_domain_var)
// {
//     Node save_mesh, load_mesh, info;
//     const int ndomains = 4;
//     blueprint::mesh::examples::spiral(ndomains, save_mesh);

//     // remove information for a particular domain
//     save_mesh[2]["fields"].remove_child("dist");

//     const std::string basename = "silo_missing_domain_var_spiral";
//     const std::string filename = basename + ".cycle_000000.root";

//     remove_path_if_exists(filename);
//     io::silo::save_mesh(save_mesh, basename);
//     io::silo::load_mesh(filename, load_mesh);

//     EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

//     // make changes to save mesh so the diff will pass
//     for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
//     {
//         silo_name_changer("mesh", save_mesh[child]);
//         int cycle = save_mesh[child]["state"]["cycle"].as_int32();
//         save_mesh[child]["state"]["cycle"].reset();
//         save_mesh[child]["state"]["cycle"] = (int64) cycle;
//     }
//     save_mesh[2].remove_child("fields");

//     EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
//     NodeConstIterator l_itr = load_mesh.children();
//     NodeConstIterator s_itr = save_mesh.children();
//     while (l_itr.has_next())
//     {
//         const Node &l_curr = l_itr.next();
//         const Node &s_curr = s_itr.next();

//         EXPECT_FALSE(l_curr.diff(s_curr, info));
//     }
// }

// //-----------------------------------------------------------------------------
// // mesh is not defined on a domain
// // 
// // This case is much less interesting.
// // data passes through the clean mesh filter which
// // deletes domains that are missing topos.
// // They simply are not part of the mesh and so silo 
// // doesn't have to deal with it.
// TEST(conduit_relay_io_silo, missing_domain_mesh_trivial)
// {
//     Node save_mesh, load_mesh, info;
//     const int ndomains = 4;
//     blueprint::mesh::examples::spiral(ndomains, save_mesh);

//     // remove information for a particular domain
//     save_mesh[2]["topologies"].remove_child("topo");

//     const std::string basename = "silo_missing_domain_mesh_trivial_spiral";
//     const std::string filename = basename + ".cycle_000000.root";

//     remove_path_if_exists(filename);
//     io::silo::save_mesh(save_mesh, basename);
//     io::silo::load_mesh(filename, load_mesh);

//     EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

//     // make changes to save mesh so the diff will pass
//     save_mesh.remove(2);
//     save_mesh.rename_child("domain_000003", "domain_000002");
//     save_mesh[2]["state"]["domain_id"].reset();
//     save_mesh[2]["state"]["domain_id"] = 2;
//     for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
//     {
//         silo_name_changer("mesh", save_mesh[child]);
//         int cycle = save_mesh[child]["state"]["cycle"].as_int32();
//         save_mesh[child]["state"]["cycle"].reset();
//         save_mesh[child]["state"]["cycle"] = (int64) cycle;
//     }

//     EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
//     NodeConstIterator l_itr = load_mesh.children();
//     NodeConstIterator s_itr = save_mesh.children();
//     while (l_itr.has_next())
//     {
//         const Node &l_curr = l_itr.next();
//         const Node &s_curr = s_itr.next();

//         EXPECT_FALSE(l_curr.diff(s_curr, info));
//     }
// }

// //-----------------------------------------------------------------------------
// // mesh is not defined on a domain but there are multiple meshes
// TEST(conduit_relay_io_silo, missing_domain_mesh)
// {
//     Node save_mesh, save_mesh2, load_mesh, load_mesh2, info, opts;
//     const int ndomains = 4;
//     blueprint::mesh::examples::spiral(ndomains, save_mesh);
//     blueprint::mesh::examples::spiral(ndomains, save_mesh2);

//     for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
//     {
//         save_mesh[child]["coordsets"].rename_child("coords", "coords2");
//         save_mesh[child]["topologies"]["topo"]["coordset"].reset();
//         save_mesh[child]["topologies"]["topo"]["coordset"] = "coords2";
//         save_mesh[child]["topologies"].rename_child("topo", "topo2");
//         save_mesh[child]["fields"]["dist"]["topology"].reset();
//         save_mesh[child]["fields"]["dist"]["topology"] = "topo2";
//         save_mesh[child]["fields"].rename_child("dist", "dist2");

//         save_mesh[child]["coordsets"]["coords"].set_external(save_mesh2[child]["coordsets"]["coords"]);
//         save_mesh[child]["topologies"]["topo"].set_external(save_mesh2[child]["topologies"]["topo"]);
//         save_mesh[child]["fields"]["dist"].set_external(save_mesh2[child]["fields"]["dist"]);
//     }

//     // remove information for a particular domain
//     save_mesh[2]["topologies"].remove_child("topo");

//     const std::string basename = "silo_missing_domain_mesh_spiral";
//     const std::string filename = basename + ".cycle_000000.root";

//     remove_path_if_exists(filename);
//     io::silo::save_mesh(save_mesh, basename);
    
//     opts["mesh_name"] = "mesh_topo2";
//     io::silo::load_mesh(filename, opts, load_mesh);
//     opts["mesh_name"] = "mesh_topo";
//     io::silo::load_mesh(filename, opts, load_mesh2);

//     EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));
//     EXPECT_TRUE(blueprint::mesh::verify(load_mesh2, info));

//     // make changes to save mesh so the diff will pass
//     save_mesh[2]["coordsets"].remove_child("coords");
//     save_mesh[2]["fields"].remove_child("dist");
//     for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
//     {
//         silo_name_changer("mesh", save_mesh[child]);
//         int cycle = save_mesh[child]["state"]["cycle"].as_int32();
//         save_mesh[child]["state"]["cycle"].reset();
//         save_mesh[child]["state"]["cycle"] = (int64) cycle;
//     }

//     // we must merge the two meshes in load mesh
//     // this is tricky because one is missing a domain
//     load_mesh[0]["coordsets"]["mesh_topo"].set_external(load_mesh2[0]["coordsets"]["mesh_topo"]);
//     load_mesh[0]["topologies"]["mesh_topo"].set_external(load_mesh2[0]["topologies"]["mesh_topo"]);
//     load_mesh[0]["fields"]["mesh_dist"].set_external(load_mesh2[0]["fields"]["mesh_dist"]);
//     load_mesh[1]["coordsets"]["mesh_topo"].set_external(load_mesh2[1]["coordsets"]["mesh_topo"]);
//     load_mesh[1]["topologies"]["mesh_topo"].set_external(load_mesh2[1]["topologies"]["mesh_topo"]);
//     load_mesh[1]["fields"]["mesh_dist"].set_external(load_mesh2[1]["fields"]["mesh_dist"]);
//     load_mesh[3]["coordsets"]["mesh_topo"].set_external(load_mesh2[2]["coordsets"]["mesh_topo"]);
//     load_mesh[3]["topologies"]["mesh_topo"].set_external(load_mesh2[2]["topologies"]["mesh_topo"]);
//     load_mesh[3]["fields"]["mesh_dist"].set_external(load_mesh2[2]["fields"]["mesh_dist"]);

//     EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
//     NodeConstIterator l_itr = load_mesh.children();
//     NodeConstIterator s_itr = save_mesh.children();
//     while (l_itr.has_next())
//     {
//         const Node &l_curr = l_itr.next();
//         const Node &s_curr = s_itr.next();

//         EXPECT_FALSE(l_curr.diff(s_curr, info));
//     }
// }

// //-----------------------------------------------------------------------------
// // explicit points (unstructured mesh) do not use every coord
// TEST(conduit_relay_io_silo, unstructured_points)
// {
//     Node save_mesh, load_mesh, info;
//     blueprint::mesh::examples::braid("points", 2, 2, 2, save_mesh);

//     std::vector<int> new_conn;
//     std::vector<float> new_field1;
//     std::vector<float> new_field2;
//     std::vector<float64> new_xcoords, new_ycoords, new_zcoords;

//     int_accessor conn = save_mesh["topologies"]["mesh"]["elements"]["connectivity"].value();

//     float_accessor field1 = save_mesh["fields"]["braid"]["values"].value();
//     float_accessor field2 = save_mesh["fields"]["radial"]["values"].value();

//     float_accessor xcoords = save_mesh["coordsets"]["coords"]["values"]["x"].value();
//     float_accessor ycoords = save_mesh["coordsets"]["coords"]["values"]["y"].value();
//     float_accessor zcoords = save_mesh["coordsets"]["coords"]["values"]["z"].value();

//     for (int i = 1; i < conn.number_of_elements(); i += 2)
//     {
//         new_conn.push_back(conn[i]);
//         new_field1.push_back(field1[i]);
//         new_field2.push_back(field2[i]);

//         new_xcoords.push_back(xcoords[conn[i]]);
//         new_ycoords.push_back(ycoords[conn[i]]);
//         new_zcoords.push_back(zcoords[conn[i]]);
//     }
//     save_mesh["topologies"]["mesh"]["elements"]["connectivity"].reset();
//     save_mesh["topologies"]["mesh"]["elements"]["connectivity"].set(new_conn);

//     save_mesh["fields"].remove_child("vel");
//     save_mesh["fields"]["braid"]["values"].reset();
//     save_mesh["fields"]["braid"]["values"].set(new_field1);
//     save_mesh["fields"]["radial"]["values"].reset();
//     save_mesh["fields"]["radial"]["values"].set(new_field2);

//     // we have modified braid such that it only uses half of the points in the coordset

//     const std::string basename = "silo_unstructured_points_braid";
//     const std::string filename = basename + ".cycle_000100.root";

//     // remove existing root file, directory and any output files
//     remove_path_if_exists(filename);

//     io::silo::save_mesh(save_mesh, basename);
//     io::silo::load_mesh(filename, load_mesh);
//     EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//     // now we must remove the unused points and change to an implicit points topo so that the diff passes
//     save_mesh["coordsets"]["coords"]["values"]["x"].reset();
//     save_mesh["coordsets"]["coords"]["values"]["x"].set(new_xcoords);
//     save_mesh["coordsets"]["coords"]["values"]["y"].reset();
//     save_mesh["coordsets"]["coords"]["values"]["y"].set(new_ycoords);
//     save_mesh["coordsets"]["coords"]["values"]["z"].reset();
//     save_mesh["coordsets"]["coords"]["values"]["z"].set(new_zcoords);

//     save_mesh["topologies"].remove_child("mesh");
//     save_mesh["topologies"]["mesh"]["type"] = "points";
//     save_mesh["topologies"]["mesh"]["coordset"] = "coords";

//     // the association doesn't matter for point meshes
//     // we choose vertex by convention
//     save_mesh["fields"]["radial"]["association"].reset();
//     save_mesh["fields"]["radial"]["association"] = "vertex";

//     silo_name_changer("mesh", save_mesh);
//     int cycle = save_mesh["state"]["cycle"].as_uint64();
//     save_mesh["state"]["cycle"].reset();
//     save_mesh["state"]["cycle"] = (int64) cycle;

//     // the loaded mesh will be in the multidomain format
//     // but the saved mesh is in the single domain format
//     EXPECT_EQ(load_mesh.number_of_children(), 1);
//     EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

//     EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
// }

// //-----------------------------------------------------------------------------

// // 
// // save and read option tests
// // 

// // save options:
// /// opts:
// ///
// ///      file_style: "default", "root_only", "multi_file", "overlink"
// ///            when # of domains == 1,  "default"   ==> "root_only"
// ///            else,                    "default"   ==> "multi_file"
// ///
// ///      silo_type: "default", "pdb", "hdf5", "unknown"
// ///            when the file we are writing to exists, "default" ==> "unknown"
// ///            else,                                   "default" ==> "hdf5"
// ///         note: these are additional silo_type options that we could add 
// ///         support for in the future:
// ///           "hdf5_sec2", "hdf5_stdio", "hdf5_mpio", "hdf5_mpiposix", "taurus"
// ///
// ///      suffix: "default", "cycle", "none"
// ///            when cycle is present,  "default"   ==> "cycle"
// ///            else,                   "default"   ==> "none"
// ///
// ///      mesh_name:  (used if present, default ==> "mesh")
// ///
// ///      ovl_topo_name: (used if present, default ==> "")
// ///
// ///      number_of_files:  {# of files}
// ///            when "multi_file" or "overlink":
// ///                 <= 0, use # of files == # of domains
// ///                  > 0, # of files == number_of_files

// // read options:
// /// opts:
// ///      mesh_name: "{name}"
// ///          provide explicit mesh name, for cases where silo data includes
// ///           more than one mesh.

// //-----------------------------------------------------------------------------
// TEST(conduit_relay_io_silo, round_trip_save_option_file_style)
// {
//     // we will do overlink tests separately
//     const std::vector<std::string> file_styles = {"default", "root_only", "multi_file"};
//     for (int i = 0; i < file_styles.size(); i ++)
//     {
//         Node opts;
//         opts["file_style"] = file_styles[i];

//         const std::string basename = "silo_save_option_file_style_" + file_styles[i] + "_spiral";
//         const std::string filename = basename + ".cycle_000000.root";

//         for (int ndomains = 1; ndomains < 5; ndomains += 3)
//         {
//             Node save_mesh, load_mesh, info;
//             blueprint::mesh::examples::spiral(ndomains, save_mesh);
//             remove_path_if_exists(filename);
//             io::silo::save_mesh(save_mesh, basename, opts);
//             io::silo::load_mesh(filename, load_mesh);
//             EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

//             // make changes to save mesh so the diff will pass
//             for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
//             {
//                 silo_name_changer("mesh", save_mesh[child]);
//                 int cycle = save_mesh[child]["state"]["cycle"].as_int32();
//                 save_mesh[child]["state"]["cycle"].reset();
//                 save_mesh[child]["state"]["cycle"] = (int64) cycle;
//             }

//             EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
//             NodeConstIterator l_itr = load_mesh.children();
//             NodeConstIterator s_itr = save_mesh.children();
//             while (l_itr.has_next())
//             {
//                 const Node &l_curr = l_itr.next();
//                 const Node &s_curr = s_itr.next();

//                 EXPECT_FALSE(l_curr.diff(s_curr, info));
//             }
//         }
//     }
// }

// //-----------------------------------------------------------------------------
// TEST(conduit_relay_io_silo, round_trip_save_option_number_of_files)
// {
//     const std::vector<int> number_of_files = {-1, 2};
//     for (int i = 0; i < number_of_files.size(); i ++)
//     {
//         Node opts;
//         opts["file_style"] = "multi_file";
//         opts["number_of_files"] = number_of_files[i];

//         const std::string basename = "silo_save_option_number_of_files_" + 
//                                      std::to_string(number_of_files[i]) + 
//                                      "_spiral";
//         const std::string filename = basename + ".cycle_000000.root";

//         int ndomains = 5;

//         Node save_mesh, load_mesh, info;
//         blueprint::mesh::examples::spiral(ndomains, save_mesh);
//         remove_path_if_exists(filename);
//         io::silo::save_mesh(save_mesh, basename, opts);
//         io::silo::load_mesh(filename, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

//         // make changes to save mesh so the diff will pass
//         for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
//         {
//             silo_name_changer("mesh", save_mesh[child]);
//             int cycle = save_mesh[child]["state"]["cycle"].as_int32();
//             save_mesh[child]["state"]["cycle"].reset();
//             save_mesh[child]["state"]["cycle"] = (int64) cycle;
//         }

//         EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
//         NodeConstIterator l_itr = load_mesh.children();
//         NodeConstIterator s_itr = save_mesh.children();
//         while (l_itr.has_next())
//         {
//             const Node &l_curr = l_itr.next();
//             const Node &s_curr = s_itr.next();

//             EXPECT_FALSE(l_curr.diff(s_curr, info));
//         }
//     }
// }

// //-----------------------------------------------------------------------------
// TEST(conduit_relay_io_silo, round_trip_save_option_suffix)
// {
//     const std::vector<std::string> suffixes = {"default", "default", "cycle", "none"};
//     const std::vector<std::string> file_suffixes = {
//         "",              // cycle is not present
//         ".cycle_000005", // cycle is present
//         ".cycle_000005", // cycle is turned on
//         "",              // cycle is turned off
//     };
//     const std::vector<std::string> include_cycle = {"no", "yes", "yes", "yes"};
//     for (int i = 0; i < suffixes.size(); i ++)
//     {
//         Node opts;
//         opts["suffix"] = suffixes[i];

//         const std::string basename = "silo_save_option_suffix_" + suffixes[i] +
//                                      "_" + include_cycle[i] + "_basic";
//         const std::string filename = basename + file_suffixes[i] + ".root";

//         Node save_mesh, load_mesh, info;
//         blueprint::mesh::examples::basic("rectilinear", 3, 4, 0, save_mesh);

//         if (include_cycle[i] == "yes")
//         {
//             save_mesh["state/cycle"] = (int64) 5;
//         }

//         remove_path_if_exists(filename);
//         io::silo::save_mesh(save_mesh, basename, opts);
//         io::silo::load_mesh(filename, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         // this is to pass the diff, as silo will add cycle in if it is not there
//         if (include_cycle[i] == "no")
//         {
//             save_mesh["state/cycle"] = (int64) 0;
//         }
//         save_mesh["state/domain_id"] = 0;
//         silo_name_changer("mesh", save_mesh);

//         // the loaded mesh will be in the multidomain format
//         // but the saved mesh is in the single domain format
//         EXPECT_EQ(load_mesh.number_of_children(), 1);
//         EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

//         EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
//     }
// }

// //-----------------------------------------------------------------------------
// TEST(conduit_relay_io_silo, round_trip_save_option_mesh_name)
// {
//     const std::string basename = "silo_save_option_mesh_name_basic";
//     const std::string filename = basename + ".root";

//     Node opts;
//     opts["mesh_name"] = "mymesh";

//     Node save_mesh, load_mesh, info;
//     blueprint::mesh::examples::basic("rectilinear", 3, 4, 0, save_mesh);
//     remove_path_if_exists(filename);
//     io::silo::save_mesh(save_mesh, basename, opts);
//     io::silo::load_mesh(filename, load_mesh);
//     EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//     save_mesh["state/cycle"] = (int64) 0;
//     save_mesh["state/domain_id"] = 0;
//     silo_name_changer("mymesh", save_mesh);

//     // the loaded mesh will be in the multidomain format
//     // but the saved mesh is in the single domain format
//     EXPECT_EQ(load_mesh.number_of_children(), 1);
//     EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());
//     EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
// }

// //-----------------------------------------------------------------------------
// TEST(conduit_relay_io_silo, round_trip_read_option_mesh_name)
// {
//     Node load_mesh, info, opts;
//     std::string path = utils::join_file_path("silo", "multi_curv3d.silo");
//     std::string input_file = relay_test_silo_data_path(path);

//     opts["mesh_name"] = "mesh1_dup";

//     io::silo::load_mesh(input_file, opts, load_mesh);
//     EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//     EXPECT_TRUE(load_mesh[0].has_path("topologies/mesh1_dup"));
// }

// //-----------------------------------------------------------------------------
// TEST(conduit_relay_io_silo, round_trip_save_option_silo_type)
// {
//     const std::vector<std::string> silo_types = {"default", "pdb", "hdf5", "unknown"};
//     for (int i = 3; i < silo_types.size(); i ++)
//     {
//         Node opts;
//         opts["silo_type"] = silo_types[i];

//         Node save_mesh, load_mesh, info;
//         blueprint::mesh::examples::basic("rectilinear", 3, 4, 0, save_mesh);

//         const std::string basename = "silo_save_option_silo_type_" + silo_types[i] + "_basic";
//         const std::string filename = basename + ".root";

//         remove_path_if_exists(filename);
//         io::silo::save_mesh(save_mesh, basename, opts);
//         io::silo::load_mesh(filename, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         // this is to pass the diff, as silo will add cycle in if it is not there
//         save_mesh["state/cycle"] = (int64) 0;
//         save_mesh["state/domain_id"] = 0;
        
//         silo_name_changer("mesh", save_mesh);

//         // the loaded mesh will be in the multidomain format
//         // but the saved mesh is in the single domain format
//         EXPECT_EQ(load_mesh.number_of_children(), 1);
//         EXPECT_EQ(load_mesh[0].number_of_children(), save_mesh.number_of_children());

//         EXPECT_FALSE(load_mesh[0].diff(save_mesh, info));
//     }
// }

// //-----------------------------------------------------------------------------
// TEST(conduit_relay_io_silo, round_trip_save_option_overlink)
// {
//     const std::vector<std::string> ovl_topo_names = {"", "topo"};
//     for (int i = 0; i < ovl_topo_names.size(); i ++)
//     {
//         std::string basename;
//         if (ovl_topo_names[i].empty())
//         {
//             basename = "silo_save_option_overlink_spiral";
//         }
//         else
//         {
//             basename = "silo_save_option_overlink_spiral_" + ovl_topo_names[i];
//         }
//         const std::string filename = basename + "/OvlTop.silo";

//         Node opts;
//         opts["file_style"] = "overlink";
//         opts["ovl_topo_name"] = ovl_topo_names[i];

//         int ndomains = 2;

//         Node save_mesh, load_mesh, info;
//         blueprint::mesh::examples::spiral(ndomains, save_mesh);
//         remove_path_if_exists(basename);
//         io::silo::save_mesh(save_mesh, basename, opts);
//         io::silo::load_mesh(filename, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh,info));

//         for (index_t child = 0; child < save_mesh.number_of_children(); child ++)
//         {
//             overlink_name_changer(save_mesh[child]);
//             int cycle = save_mesh[child]["state"]["cycle"].as_int32();
//             save_mesh[child]["state"]["cycle"].reset();
//             save_mesh[child]["state"]["cycle"] = (int64) cycle;
//         }

//         EXPECT_EQ(load_mesh.number_of_children(), save_mesh.number_of_children());
//         NodeConstIterator l_itr = load_mesh.children();
//         NodeConstIterator s_itr = save_mesh.children();
//         while (l_itr.has_next())
//         {
//             const Node &l_curr = l_itr.next();
//             const Node &s_curr = s_itr.next();

//             EXPECT_FALSE(l_curr.diff(s_curr, info));
//         }
//     }
// }

// //-----------------------------------------------------------------------------

// //
// // read and write Silo and Overlink tests
// //

// //-----------------------------------------------------------------------------
// // read normal silo files containing multimeshes and multivars
// TEST(conduit_relay_io_silo, read_silo)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//         {".",                  "multi_curv3d", ".silo", ""            }, // test default case
//         {".",                  "multi_curv3d", ".silo", "mesh1"       },
//         {".",                  "multi_curv3d", ".silo", "mesh1_back"  },
//         {".",                  "multi_curv3d", ".silo", "mesh1_dup"   },
//         {".",                  "multi_curv3d", ".silo", "mesh1_front" },
//         {".",                  "multi_curv3d", ".silo", "mesh1_hidden"},
//         {".",                  "tire",         ".silo", ""            }, // test default case
//         {".",                  "tire",         ".silo", "tire"        },
//         {".",                  "galaxy0000",   ".silo", ""            }, // test default case
//         {".",                  "galaxy0000",   ".silo", "StarMesh"    },
//         {".",                  "emptydomains", ".silo", ""            }, // test default case
//         {".",                  "emptydomains", ".silo", "mesh"        },
//         {"multidir_test_data", "multidir0000", ".root", ""            }, // test default case
//         {"multidir_test_data", "multidir0000", ".root", "Mesh"        },
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;
//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("silo", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_silo_" + basename;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = meshname;
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// //-----------------------------------------------------------------------------
// // test that we can read the fake overlink files from the visit test data
// TEST(conduit_relay_io_silo, read_fake_overlink)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//      // {"ev_0_0_100",              "OvlTop", ".silo", ""     }, // test default case
//      // {"ev_0_0_100",              "OvlTop", ".silo", "MMESH"},
//         // uncomment once silo ucdmesh phzones are supported
//         {"hl18spec",                "OvlTop", ".silo", ""     }, // test default case
//         {"hl18spec",                "OvlTop", ".silo", "MMESH"},
//      // {"regrovl_qh_1000_10001_4", "OvlTop", ".silo", ""     }, // test default case
//      // {"regrovl_qh_1000_10001_4", "OvlTop", ".silo", "MMESH"},
//         // uncomment once silo ucdmesh phzones are supported
//         {"utpyr4",                  "OvlTop", ".silo", ""     }, // test default case
//         {"utpyr4",                  "OvlTop", ".silo", "MMESH"},
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;
//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("fake_overlink", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_fake_overlink_" + dirname;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = "MMESH";
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// //-----------------------------------------------------------------------------
// // read overlink files in symlink format
// // should be similar to reading raw silo
// TEST(conduit_relay_io_silo, read_overlink_symlink_format)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//         {".", "box2d",                  ".silo", ""     }, // test default case
//         {".", "box2d",                  ".silo", "MMESH"},
//         {".", "box3d",                  ".silo", ""     }, // test default case
//         {".", "box3d",                  ".silo", "MMESH"},
//      // {".", "diamond",                ".silo", ""     }, // test default case
//      // {".", "diamond",                ".silo", "MMESH"},
//         // fails b/c polytopal not yet supported
//         {".", "testDisk2D_a",           ".silo", ""     }, // test default case
//         {".", "testDisk2D_a",           ".silo", "MMESH"},
//      // {".", "donordiv.s2_materials2", ".silo", ""     }, // test default case
//      // {".", "donordiv.s2_materials2", ".silo", "MMESH"},
//         // fails b/c polytopal not yet supported
//         {".", "donordiv.s2_materials3", ".silo", ""     }, // test default case
//         {".", "donordiv.s2_materials3", ".silo", "MMESH"},
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;
//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("overlink", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_overlink_symlink_" + basename;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = "MMESH";
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// //-----------------------------------------------------------------------------
// // read overlink directly from ovltop.silo
// // this case is tricky and involves messing with paths
// TEST(conduit_relay_io_silo, read_overlink_directly)
// {
//     const std::vector<std::vector<std::string>> file_info = {
//         {"box2d",                  "OvlTop", ".silo", ""     }, // test default case
//         {"box2d",                  "OvlTop", ".silo", "MMESH"},
//         {"box3d",                  "OvlTop", ".silo", ""     }, // test default case
//         {"box3d",                  "OvlTop", ".silo", "MMESH"},
//      // {"diamond",                "OvlTop", ".silo", ""     }, // test default case
//      // {"diamond",                "OvlTop", ".silo", "MMESH"},
//         {"testDisk2D_a",           "OvlTop", ".silo", ""     }, // test default case
//         {"testDisk2D_a",           "OvlTop", ".silo", "MMESH"},
//      // {"donordiv.s2_materials2", "OvlTop", ".silo", ""     }, // test default case
//      // {"donordiv.s2_materials2", "OvlTop", ".silo", "MMESH"},
//         {"donordiv.s2_materials3", "OvlTop", ".silo", ""     }, // test default case
//         {"donordiv.s2_materials3", "OvlTop", ".silo", "MMESH"},
//     };

//     for (int i = 0; i < file_info.size(); i ++) 
//     {
//         const std::string dirname  = file_info[i][0];
//         const std::string basename = file_info[i][1];
//         const std::string fileext  = file_info[i][2];
//         const std::string meshname = file_info[i][3];

//         Node load_mesh, info, read_opts, write_opts;

//         std::string filepath = utils::join_file_path(dirname, basename) + fileext;
//         filepath = utils::join_file_path("overlink", filepath);
//         std::string input_file = relay_test_silo_data_path(filepath);

//         read_opts["mesh_name"] = meshname;

//         io::silo::load_mesh(input_file, read_opts, load_mesh);
//         EXPECT_TRUE(blueprint::mesh::verify(load_mesh, info));

//         std::string out_name = "read_overlink_direct_" + dirname;
//         if (!meshname.empty())
//         {
//             out_name += "_" + meshname;
//         }

//         remove_path_if_exists(out_name + "_write_blueprint");
//         io::blueprint::save_mesh(load_mesh, out_name + "_write_blueprint", "hdf5");

//         remove_path_if_exists(out_name + "_write_silo");
//         io::silo::save_mesh(load_mesh, out_name + "_write_silo");

//         // TODO uncomment when overlink is fully supported
//         // remove_path_if_exists(out_name + "_write_overlink");
//         // write_opts["file_style"] = "overlink";
//         // write_opts["ovl_topo_name"] = "MMESH";
//         // io::silo::save_mesh(load_mesh, out_name + "_write_overlink", write_opts);
//     }
// }

// // TODO add tests for...
// //  - materials once they are supported
// //  - polytopal meshes once they are supported
// //  - units once they are supported
// //  - etc.

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
