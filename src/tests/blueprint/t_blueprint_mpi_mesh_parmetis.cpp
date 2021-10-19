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
TEST(blueprint_mpi_parmetis, basic)
{
    int par_size, par_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);


    // test with a 2d poly example
    index_t nlevels = 2;
    index_t nz = 1;
    Node mesh, side_mesh, info;

    // create polytessalation with two levels
    conduit::blueprint::mesh::examples::polytess(nlevels, nz, mesh.append());

    // on the second mpi task, shift all the coords over,
    // so domains don't overlap
    if(par_rank == 1)
    {
        float64_array xvals = mesh[0]["coordsets/coords/values/x"].value();

        for(index_t i=0;i<xvals.number_of_elements();i++)
        {
            xvals[i] += 6.0;
        }
    }
    
    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    // paint a field with parmetis result (WIP)
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,MPI_COMM_WORLD);

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/topo"];
    Node &side_fields = side_mesh["fields"];
    
    // we can't map vert assoced fields yet
    Node opts;
    opts["field_names"].append().set("global_element_ids");
    opts["field_names"].append().set("parmetis_result");

    // gen sides and save so we can look at this in visit.
    blueprint::mesh::topology::unstructured::generate_sides(mesh[0]["topologies/topo"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            opts);


    std::string output_base = "tout_bp_mpi_mesh_parametis_poly2d_test";

    // Node opts;
    // opts["file_style"] = "root_only";
    conduit::relay::mpi::io::blueprint::save_mesh(side_mesh,
                                                  output_base,
                                                  "hdf5",
                                                   // opts,
                                                   MPI_COMM_WORLD);
    EXPECT_TRUE(true);

}

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_parmetis, braid)
{
    int par_size, par_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);

    const int npts = 10;

    // test with a 2d poly example
    Node mesh, side_mesh, info;

    // create braid
    conduit::blueprint::mesh::examples::braid("structured",
                                              npts, npts, 1,
                                              mesh.append());
    Node unstruct_topo, unstruct_coords;
    conduit::blueprint::mesh::topology::structured::to_unstructured(mesh[0]["topologies/mesh"],
                                                                    unstruct_topo,
                                                                    unstruct_coords);
    Node unstruct_topo_poly;
    conduit::blueprint::mesh::topology::unstructured::to_polygonal(unstruct_topo, unstruct_topo_poly);
    mesh[0]["topologies/mesh"] = unstruct_topo_poly;
    mesh[0]["topologies/mesh/coordset"] = "coords";
    mesh[0]["coordsets/coords"] = unstruct_coords;

    // the example data set has the bounds -10 to 10 in all dims
    // Offset this along x to create mpi 'pencil'
    if(par_rank > 0)
    {
        float64_array xvals = mesh[0]["coordsets/coords/values/x"].value();

        for(index_t i=0;i<xvals.number_of_elements();i++)
        {
            xvals[i] += 21.0 * par_rank;
        }
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node options;
    options["topology"] = "mesh";

    // paint a field with parmetis result (WIP)
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,options,MPI_COMM_WORLD);

    Node s2dmap, d2smap;
    Node &side_coords = side_mesh["coordsets/coords"];
    Node &side_topo = side_mesh["topologies/mesh"];
    Node &side_fields = side_mesh["fields"];

    // we can't map vert assoced fields yet
    Node opts;
    opts["field_names"].append().set("global_element_ids");
    opts["field_names"].append().set("parmetis_result");

    // gen sides and save so we can look at this in visit.
    blueprint::mesh::topology::unstructured::generate_sides(mesh[0]["topologies/mesh"],
                                                            side_topo,
                                                            side_coords,
                                                            side_fields,
                                                            s2dmap,
                                                            d2smap,
                                                            opts);

    std::string output_base = "tout_bp_mpi_mesh_parametis_braid2d_test";

    // Node opts;
    // opts["file_style"] = "root_only";
    conduit::relay::mpi::io::blueprint::save_mesh(side_mesh,
            output_base,
            "hdf5",
            // opts,
            MPI_COMM_WORLD);
    EXPECT_TRUE(true);
}

//-----------------------------------------------------------------------------
TEST(blueprint_mpi_parmetis, uniform_adjset)
{
    int par_size, par_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);

    const int npts = 10;

    // test with a 2d poly example
    Node mesh, side_mesh, info;

    // create uniform mesh with adjsets
    conduit::blueprint::mesh::examples::adjset_uniform(mesh);

    // extract local domains
    Node local_mesh;
    for (int i = 0; i < 8; i++)
    {
        if (i % par_size != par_rank)
        {
            continue;
        }
        std::ostringstream oss;
        oss << "domain_" << std::setfill('0') << std::setw(6) << i;
        const std::string domain_name = oss.str();
        local_mesh[domain_name] = mesh[domain_name];
    }

    // construct unstructured topology
    for (Node& domain : local_mesh.children())
    {
        Node unstruct_topo, unstruct_coords;
        conduit::blueprint::mesh::topology::uniform::to_unstructured(domain["topologies/topo"],
                                                                     unstruct_topo,
                                                                     unstruct_coords);
        Node unstruct_topo_poly;
        conduit::blueprint::mesh::topology::unstructured::to_polygonal(unstruct_topo, unstruct_topo_poly);
        domain["topologies/topo"] = unstruct_topo_poly;
        domain["topologies/topo/coordset"] = "coords";
        domain["coordsets/coords"] = unstruct_coords;
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(local_mesh, info));

    // paint a field with parmetis result (WIP)
    conduit::blueprint::mpi::mesh::generate_partition_field(local_mesh,MPI_COMM_WORLD);

    for (int i = 0; i < 8; i++)
    {
        if (i % par_size != par_rank)
        {
            continue;
        }
        std::ostringstream oss;
        oss << "domain_" << std::setfill('0') << std::setw(6) << i;
        const std::string domain_name = oss.str();

        const Node& in_domain = local_mesh[domain_name];
        Node& side_mesh_dom = side_mesh[domain_name];

        Node s2dmap, d2smap;
        Node &side_coords = side_mesh_dom["coordsets/coords"];
        Node &side_topo = side_mesh_dom["topologies/topo"];
        Node &side_fields = side_mesh_dom["fields"];

        // we can't map vert assoced fields yet
        Node opts;
        opts["field_names"].append().set("global_element_ids");
        opts["field_names"].append().set("parmetis_result");

        // gen sides and save so we can look at this in visit.
        blueprint::mesh::topology::unstructured::generate_sides(in_domain["topologies/topo"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                opts);
    }
    std::string output_base = "tout_bp_mpi_mesh_parametis_uniform_adjset2d_test";

    // Node opts;
    // opts["file_style"] = "root_only";
    conduit::relay::mpi::io::blueprint::save_mesh(side_mesh,
            output_base,
            "hdf5",
            // opts,
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
