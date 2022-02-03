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
    Node part_opts;
    part_opts["partitions"] = 2;
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,
                                                            part_opts,
                                                            MPI_COMM_WORLD);

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
    Node mesh, info;

    // create braid - one domain per rank
    conduit::blueprint::mesh::examples::braid("structured",
                                              npts, npts, 1,
                                              mesh.append());
    Node unstruct_topo, unstruct_coords;
    conduit::blueprint::mesh::topology::structured::to_unstructured(mesh[0]["topologies/mesh"],
                                                                    unstruct_topo,
                                                                    unstruct_coords);
    Node unstruct_topo_poly;
    conduit::blueprint::mesh::topology::unstructured::to_polygonal(unstruct_topo, unstruct_topo_poly);
    mesh[0]["state/domain_id"] = par_rank;
    mesh[0]["topologies/mesh"] = unstruct_topo_poly;
    mesh[0]["topologies/mesh/coordset"] = "coords";
    mesh[0]["coordsets/coords"] = unstruct_coords;

    if (par_size > 1)
    {
        // Construct adjsets for multi-domain case
        const Node& dom_cset = mesh[0]["coordsets/coords"];
        float64_array coord_array[2];
        coord_array[0] = dom_cset["values/x"].as_float64_array();
        coord_array[1] = dom_cset["values/y"].as_float64_array();

        // Points with x = -10 are shared with rank - 1
        // Points with x = 10 are shared with rank + 1
        std::vector<index_t> prev_rank_shared;
        std::vector<index_t> next_rank_shared;
        for (index_t i = 0; i < coord_array[0].number_of_elements(); i++)
        {
            if (coord_array[0][i] == -10.0)
            {
                prev_rank_shared.push_back(i);
            }
            else if (coord_array[0][i] == 10)
            {
                next_rank_shared.push_back(i);
            }
        }
        // Sort by corresponding y-index to ensure common ordering of shared
        // vertex ids
        std::sort(prev_rank_shared.begin(), prev_rank_shared.end(),
                  [&] (index_t a, index_t b)
                  { return coord_array[1][a] < coord_array[1][b]; });
        std::sort(next_rank_shared.begin(), next_rank_shared.end(),
                  [&] (index_t a, index_t b)
                  { return coord_array[1][a] < coord_array[1][b]; });

        // Since we have the same domains for each rank, synchronize
        // vertex-centered fields on the boundaries to avoid discontinuities.
        for (Node& field : mesh[0]["fields"].children())
        {
            auto sync_edges =
                [&prev_rank_shared, &next_rank_shared](float64_array values)
                {
                    for (int ibdr = 0; ibdr < prev_rank_shared.size(); ibdr++)
                    {
                        index_t left_edge = prev_rank_shared[ibdr];
                        index_t right_edge = next_rank_shared[ibdr];
                        float64 max_val = std::max(values[left_edge], values[right_edge]);
                        values[left_edge] = max_val;
                        values[right_edge] = max_val;
                    }
                };
            if (field["association"].as_string() == "vertex")
            {
                if (field["values"].number_of_children() == 0)
                {
                    float64_array values = field["values"].value();
                    sync_edges(values);
                }
                else
                {
                    for (Node& comp : field["values"].children())
                    {
                        float64_array values = comp.value();
                        sync_edges(values);
                    }
                }
            }
        }

        Node& dom_aset = mesh[0]["adjsets/elem_aset"];
        dom_aset["association"] = "vertex";
        dom_aset["topology"] = "mesh";
        Node& aset_grps = dom_aset["groups"];
        if (par_rank > 0)
        {
            // Add prev rank shared node
            std::string group_name
                = conduit_fmt::format("group_{}_{}", par_rank-1, par_rank);
            Node& prev_shared = aset_grps[group_name];
            prev_shared["neighbors"].set({par_rank-1});
            prev_shared["values"].set(prev_rank_shared);
        }
        if (par_rank + 1 < par_size)
        {
            // Add next rank shared node
            std::string group_name
                = conduit_fmt::format("group_{}_{}", par_rank, par_rank+1);
            Node& prev_shared = aset_grps[group_name];
            prev_shared["neighbors"].set({par_rank+1});
            prev_shared["values"].set(next_rank_shared);
        }
    }

    // the example data set has the bounds -10 to 10 in all dims
    // Offset this along x to create mpi 'pencil'
    if(par_rank > 0)
    {
        float64_array xvals = mesh[0]["coordsets/coords/values/x"].value();

        for(index_t i=0;i<xvals.number_of_elements();i++)
        {
            xvals[i] += 20.0 * par_rank;
        }
    }

    EXPECT_TRUE(conduit::blueprint::mesh::verify(mesh, info));

    Node options;
    options["partitions"] = par_size + 1;
    options["topology"] = "mesh";
    if (par_size > 1)
    {
        options["adjset"] = "elem_aset";
    }

    // paint a field with parmetis result (WIP)
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,options,MPI_COMM_WORLD);

    Node repart_mesh;
    Node partition_options;
    Node& selection = partition_options["selections"].append();
    selection["type"] = "field";
    selection["domain_id"] = "any";
    selection["field"] = "parmetis_result";
    selection["topology"] = "mesh";

    // partition the mesh with our generated result
    conduit::blueprint::mpi::mesh::partition(mesh,
                                             partition_options,
                                             repart_mesh,
                                             MPI_COMM_WORLD);

    {
        Node side_mesh_repart;
        Node s2dmap, d2smap, opts;
        opts["field_names"].append().set("global_element_ids");
        opts["field_names"].append().set("parmetis_result");
        opts["field_names"].append().set("braid");
        opts["field_names"].append().set("radial");
        //opts["field_names"].append().set("vel");
        opts["field_names"].append().set("is_shared_node");

        Node repart_mesh_multidom;
        if (!conduit::blueprint::mesh::is_multi_domain(repart_mesh))
        {
            conduit::blueprint::mesh::to_multi_domain(repart_mesh, repart_mesh_multidom);
        }
        else
        {
            repart_mesh_multidom.set_external(repart_mesh);
        }

        for (Node& dom : repart_mesh_multidom.children())
        {
            Node& shared_nodes = dom["fields/is_shared_node"];
            shared_nodes["association"] = "vertex";
            shared_nodes["type"] = "scalar";
            shared_nodes["topology"] = "mesh";
            shared_nodes["values"].set(DataType::float64(dom["fields/braid/values"].dtype().number_of_elements()));
            float64_array shared_nodes_val = shared_nodes["values"].value();
            shared_nodes_val.fill(-1);

            const Node& groups = dom["adjsets/elem_aset/groups"];
            for (const Node& group : groups.children())
            {
                int64 nbr = group["neighbors"].as_int64();
                int64_accessor grp_vals = group["values"].as_int64_accessor();
                for (index_t iv = 0; iv < grp_vals.number_of_elements(); iv++)
                {
                    shared_nodes_val[grp_vals[iv]] = double(iv) / grp_vals.number_of_elements();
                }
            }
        }

        for (const Node& dom : repart_mesh_multidom.children())
        {
            Node& side_domain = side_mesh_repart.append();

            Node &side_coords = side_domain["coordsets/coords"];
            Node &side_topo = side_domain["topologies/mesh"];
            Node &side_fields = side_domain["fields"];

            // gen sides and save so we can look at this in visit.
            blueprint::mesh::topology::unstructured::generate_sides(dom["topologies/mesh"],
                                                                    side_topo,
                                                                    side_coords,
                                                                    side_fields,
                                                                    s2dmap,
                                                                    d2smap,
                                                                    opts);
        }
        std::string output_repart_base = "tout_bp_mpi_mesh_parmetis_repart_braid2d_test";

        conduit::relay::mpi::io::blueprint::save_mesh(side_mesh_repart,
                                                      output_repart_base,
                                                      "hdf5",
                                                      MPI_COMM_WORLD);
    }

    {
        auto orig_doms = conduit::blueprint::mesh::domains(mesh);
        for (conduit::Node* pdom : orig_doms)
        {
            // Delete field we're mapping back from repartitioned mesh
            pdom->child("fields").remove("braid");
            pdom->child("fields").remove("radial");
        }
    }
    // Perform a map-back of some zone-centered variables
    conduit::blueprint::mpi::mesh::partition_map_back(repart_mesh,
                                                      mesh,
                                                      {"braid", "radial"},
                                                      MPI_COMM_WORLD);

    {
        Node side_mesh;
        Node s2dmap, d2smap;
        Node &side_coords = side_mesh["coordsets/coords"];
        Node &side_topo = side_mesh["topologies/mesh"];
        Node &side_fields = side_mesh["fields"];

        // we can't map vert assoced fields yet
        Node opts;
        opts["field_names"].append().set("global_element_ids");
        opts["field_names"].append().set("parmetis_result");
        opts["field_names"].append().set("braid");
        opts["field_names"].append().set("radial");

        // gen sides and save so we can look at this in visit.
        blueprint::mesh::topology::unstructured::generate_sides(mesh[0]["topologies/mesh"],
                                                                side_topo,
                                                                side_coords,
                                                                side_fields,
                                                                s2dmap,
                                                                d2smap,
                                                                opts);

        std::string output_base = "tout_bp_mpi_mesh_parmetis_braid2d_test_prepart";

        // Node opts;
        // opts["file_style"] = "root_only";
        conduit::relay::mpi::io::blueprint::save_mesh(side_mesh,
                                                      output_base,
                                                      "hdf5",
                                                      // opts,
                                                      MPI_COMM_WORLD);
    }

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
    Node part_opts;
    part_opts["partition"] = 3;
    conduit::blueprint::mpi::mesh::generate_partition_field(local_mesh,
                                                             part_opts,
                                                             MPI_COMM_WORLD);

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
