// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mpi_mesh_parmetis.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_mpi.hpp"
#include "conduit_blueprint_mpi_mesh_parmetis.hpp"
#include "conduit_relay.hpp"
#include "conduit_relay_mpi.hpp"
#include "conduit_relay_mpi_io.hpp"
#include "conduit_relay_mpi_io_blueprint.hpp"
#include "conduit_utils.hpp"
#include "conduit_fmt/conduit_fmt.h"

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
bool
check_if_hdf5_enabled()
{
    Node io_protos;
    relay::io::about(io_protos["io"]);
    return io_protos["io/protocols/hdf5"].as_string() == "enabled";
}

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_load_bal, basic)
{
    //

    int par_size, par_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);

    // build test using related_boundary example
    // one way to create unbalanced number of eles across ranks:
    //  domain 0 --> rank 0
    //  domain 1 and 2 --> rank 0

    Node mesh;
    index_t base_grid_ele_i = 3;
    index_t base_grid_ele_j = 3;

    // gen on all ranks, sub select domains for mpi use
    conduit::blueprint::mesh::examples::related_boundary(base_grid_ele_i,
                                                         base_grid_ele_j,
                                                         mesh);
    if(par_size > 1)
    {
        if(par_rank == 0)
        {
            // keep domains 0 and 1, remove domain 2
            mesh.remove("domain2");
        }
        else // rank == 1;
        {
            // keep domain 2, remove domains 0 and 1
            mesh.remove("domain0");
            mesh.remove("domain1");
        }
    }

    std::string output_base = "tout_bp_mpi_load_bal_basic_input";

    // prefer hdf5, fall back to yaml
    std::string protocol = "yaml";

    if(check_if_hdf5_enabled())
    {
        protocol = "hdf5";
    }

    conduit::relay::mpi::io::blueprint::save_mesh(mesh,
                                                  output_base,
                                                  protocol,
                                                  MPI_COMM_WORLD);

    Node options;
    options["partitions"] = 4; // from 3 domains to 4
    options["topology"] = "main";

    // we want to map the main topology according to parmetis
    // first lets create the parmetis dest field
    std::cout  << "gen parmetis field" << std::endl;
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,options,MPI_COMM_WORLD);

    // paint the parmetis result on the boundary mesh, using the parmetis field 
    // and the relationships between the boundary and main topo

    std::cout  << "paint parmetis field on boundary topology" << std::endl;

    // loop over domains
    NodeIterator itr = mesh.children();
    while(itr.has_next())
    {
        Node &curr_dom = itr.next();

        // mesh could be empty on some ranks
        if(curr_dom.has_child("fields"))
        {
            // get main / parmetis_result
            int64_accessor main_parmetis_result = curr_dom["fields/parmetis_result/values"].value();

            // get boundary / bndry_to_main_local
            int64_accessor bndry_to_main_local_vals = curr_dom["fields/bndry_to_main_local/values"].value();
            index_t num_bndry_ele = bndry_to_main_local_vals.number_of_elements();

            // create bndry_parmetis_result
            curr_dom["fields/bndry_parmetis_result/association"] = "element";
            curr_dom["fields/bndry_parmetis_result/topology"] = "boundary";
            curr_dom["fields/bndry_parmetis_result/values"].set(DataType::int64(num_bndry_ele));
            int64_array bndry_parmetis_result_vals = curr_dom["fields/bndry_parmetis_result/values"].value();
            for(index_t i=0;i<num_bndry_ele;i++)
            {
                bndry_parmetis_result_vals[i] = main_parmetis_result[bndry_to_main_local_vals[i]];
            }
        }
    }

    std::cout  << "saving parmetis field" << std::endl;
    conduit::relay::mpi::io::blueprint::save_mesh(mesh,
                                                  "tout_bp_mpi_load_bal_basic_part_field",
                                                  protocol,
                                                  MPI_COMM_WORLD);

    // relocate both the main and boundary topology (and all of their fields)
    Node res_main;
    Node res_bndry;

    options.reset();
    {
        Node& selection = options["selections"].append();
        selection["type"] = "field";
        selection["domain_id"] = "any";
        selection["field"] = "parmetis_result";
        selection["topology"] = "main";

        std::cout  << "partitioning the main topology" << std::endl;
        
        conduit::blueprint::mpi::mesh::partition(mesh, options, res_main, MPI_COMM_WORLD);
    }
    res_main.print();

    Node bndry = mesh;

    options.reset();
    {
        itr = bndry.children();
        while(itr.has_next())
        {
            Node &curr_dom = itr.next();
            // remove any orig fields added by the partition 
            curr_dom["fields"].remove("global_element_ids");
            curr_dom["fields"].remove("global_vertex_ids");
            curr_dom.remove("adjsets");
        }



        Node &selection = options["selections"].append();
        selection["type"] = "field";
        selection["domain_id"] = "any";
        selection["field"] = "bndry_parmetis_result";
        selection["topology"] = "boundary";

        std::cout  << "partitioning the boundary topology" << std::endl;
        
        conduit::blueprint::mpi::mesh::partition(bndry, options, res_bndry, MPI_COMM_WORLD);
    }

    std::cout  << "saving partition result main mesh" << std::endl;
    conduit::relay::mpi::io::blueprint::save_mesh(res_main,
                                                  "tout_bp_mpi_load_bal_basic_part_result_main",
                                                  protocol,
                                                  MPI_COMM_WORLD);

    std::cout  << "saving partition result boundary mesh" << std::endl;
    // TODO: This is twisted.
    conduit::relay::mpi::io::blueprint::save_mesh(res_bndry,
                                                  "tout_bp_mpi_load_bal_basic_part_result_bndry",
                                                  protocol,
                                                  MPI_COMM_WORLD);


    std::cout  << "creating new field on main topology to map back" << std::endl;

    // add a BRAND NEW field!
    itr = res_main.children();
    while(itr.has_next())
    {
        Node &curr_dom = itr.next();
        // create main_new_and_improved
        index_t num_main_eles = curr_dom["fields/parmetis_result/values"].dtype().number_of_elements();
        curr_dom["fields/main_new_and_improved/association"] = "element";
        curr_dom["fields/main_new_and_improved/topology"] = "main";
        curr_dom["fields/main_new_and_improved/values"].set(DataType::int64(num_main_eles));
        int64_array main_nai_vals = curr_dom["fields/main_new_and_improved/values"].value();

        for(index_t i=0;i<num_main_eles;i++)
        {
            main_nai_vals[i] = -42 + i;
        }
    }

    std::cout  << "saving partition result mesh with new field" << std::endl;
    conduit::relay::mpi::io::blueprint::save_mesh(res_main,
                                                  "tout_bp_mpi_load_bal_basic_part_result",
                                                  protocol,
                                                  MPI_COMM_WORLD);


    Node mapback_opts;
    mapback_opts["fields"].append().set("main_new_and_improved");

    std::cout  << "mapping back new field to orignal main topology" << std::endl;
    // Perform a map-back of some zone-centered variables
    conduit::blueprint::mpi::mesh::partition_map_back(res_main,
                                                      mapback_opts,
                                                      mesh,
                                                      MPI_COMM_WORLD);

    std::cout  << "saving mapped back main result mesh" << std::endl;
    conduit::relay::mpi::io::blueprint::save_mesh(mesh,
                                                  "tout_bp_mpi_load_bal_basic_mapback_result",
                                                  protocol,
                                                  MPI_COMM_WORLD);


    EXPECT_TRUE(true);
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
