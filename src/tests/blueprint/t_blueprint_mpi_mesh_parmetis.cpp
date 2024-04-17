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

#include "blueprint_mpi_test_helpers.hpp"

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

static void print(const conduit::Node &n)
{
   conduit::Node opt;
   opt["num_children_threshold"] = 100000;
   opt["num_elements_threshold"] = 100000;
   std::cout << n.to_summary_string(opt) << std::endl;
}

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
TEST(blueprint_mpi_parmetis, polyhedra)
{
    int par_size, par_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);


    // test with a polyhedral example. Make them way unbalanced
    index_t nlevels = par_rank == 0 ? 4 : 2;
    index_t nz = 3;
    Node mesh, side_mesh, info;

    // create polyhedra
    conduit::blueprint::mesh::examples::polytess(nlevels, nz, mesh.append());

    // Figure out the translation on rank 0
    float64_array xvals = mesh[0]["coordsets/coords/values/x"].value();
    float64_array yvals = mesh[0]["coordsets/coords/values/y"].value();
    float64_array zvals = mesh[0]["coordsets/coords/values/z"].value();
    double translate[3] = {0., 0., 0.};
    if(par_rank == 0)
    {
        // These indices are good at level 4
        const int origin = 28;
        const int newOrigin = 91;
        translate[0] = xvals[newOrigin] - xvals[origin];
        translate[1] = yvals[newOrigin] - yvals[origin];
        translate[2] = zvals[newOrigin] - zvals[origin];
    }
    // Send the translation to domain 1.
    MPI_Bcast(translate, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // on the second mpi task, shift all the coords over,
    // so domains just touch.
    if(par_rank == 1)
    {
        for(index_t i=0;i<xvals.number_of_elements();i++)
        {
            xvals[i] += translate[0];
            yvals[i] += translate[1];
            zvals[i] += translate[2];
        }
    }

    // NOTE: The domain_id and the adjacency sets are critical to generating
    //       the right global vertex ids.

    // Set the domain_id.
    mesh[0]["state/domain_id"] = par_rank;

    // Add adjsets.
    const char *dom0_adjset = R"(
topo_adjset:
  association: vertex
  topology: topo
  groups:
    group_0_1:
      neighbors: 1
      values: [107,235,363,106,234,362,91,219,347,90,218,346,89,217,345,88,216,344,87,215,343,100,228,356]
)";
    const char *dom1_adjset = R"(
topo_adjset:
  association: vertex
  topology: topo
  groups:
    group_0_1:
      neighbors: 0
      values: [30,62,94,31,63,95,28,60,92,9,41,73,8,40,72,14,46,78,10,42,74,11,43,75]
)";
    mesh[0]["adjsets"].parse((par_rank == 0) ? dom0_adjset : dom1_adjset);

    bool v = conduit::blueprint::mesh::verify(mesh, info);
    if(!v)
        info.print();
    EXPECT_TRUE(v);

    // Paint a field with parmetis result
    Node part_opts;
    part_opts["partitions"] = 2;
    part_opts["adjset"] = "topo_adjset";
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,
                                                            part_opts,
                                                            MPI_COMM_WORLD);

#if 0
    // Print the results so we can make the baselines.
    in_rank_order(MPI_COMM_WORLD, [&](int rank) {
        std::cout << "Rank " << rank << std::endl;
        print(mesh[0]["fields/parmetis_result"]);
        print(mesh[0]["fields/global_vertex_ids"]);
    });
#endif

    // Baselines
    const char *dom0_parmetis_result = R"(
association: "element"
topology: "topo"
values: [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
)";
    const char *dom0_global_vertex_ids = R"(
association: "vertex"
topology: "topo"
values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383]
)";
    const char *dom1_parmetis_result = R"(
association: "element"
topology: "topo"
values: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
)";
    const char *dom1_global_vertex_ids = R"(
association: "vertex"
topology: "topo"
values: [384, 385, 386, 387, 388, 389, 390, 391, 89, 90, 87, 100, 392, 393, 88, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 91, 407, 107, 106, 408, 409, 410, 411, 412, 413, 414, 415, 217, 218, 215, 228, 416, 417, 216, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 219, 431, 235, 234, 432, 433, 434, 435, 436, 437, 438, 439, 345, 346, 343, 356, 440, 441, 344, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 347, 455, 363, 362]
)";

    // Compare fields.
    conduit::Node baselines;
    std::vector<std::string> fields{"fields/parmetis_result", "fields/global_vertex_ids"};
    if(par_rank == 0)
    {
        baselines["fields/parmetis_result"].parse(dom0_parmetis_result);
        baselines["fields/global_vertex_ids"].parse(dom0_global_vertex_ids);
    }
    else
    {
        baselines["fields/parmetis_result"].parse(dom1_parmetis_result);
        baselines["fields/global_vertex_ids"].parse(dom1_global_vertex_ids);
    }
    for(const auto &field : fields)
    {
        bool different = mesh[0][field].diff(baselines[field], info);
        in_rank_order(MPI_COMM_WORLD, [&](int rank) {
            if(different)
            {
                print(info);
            }
        });
        EXPECT_FALSE(different);
    }

    // Node opts;
    // opts["file_style"] = "root_only";
    std::string output_base = "tout_bp_mpi_mesh_parametis_polyhedra_test";
    conduit::relay::mpi::io::blueprint::save_mesh(mesh,
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
                    for (size_t ibdr = 0; ibdr < prev_rank_shared.size(); ibdr++)
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
        opts["field_names"].append().set("global_vertex_ids");
        opts["field_names"].append().set("parmetis_result");
        opts["field_names"].append().set("braid");
        opts["field_names"].append().set("radial");
        //opts["field_names"].append().set("vel");
        opts["field_names"].append().set("is_shared_node");

        auto repart_mesh_doms = conduit::blueprint::mesh::domains(repart_mesh);

        for (Node* pdom : repart_mesh_doms)
        {
            Node& dom = *pdom;
            Node& shared_nodes = dom["fields/is_shared_node"];
            shared_nodes["association"] = "vertex";
            shared_nodes["topology"] = "mesh";
            shared_nodes["values"].set(DataType::float64(dom["fields/braid/values"].dtype().number_of_elements()));
            float64_array shared_nodes_val = shared_nodes["values"].value();
            shared_nodes_val.fill(-1);

            const Node& groups = dom["adjsets/elem_aset/groups"];
            for (const Node& group : groups.children())
            {
                //int64 nbr = group["neighbors"].as_int64();
                int64_accessor grp_vals = group["values"].as_int64_accessor();
                for (index_t iv = 0; iv < grp_vals.number_of_elements(); iv++)
                {
                    shared_nodes_val[grp_vals[iv]] = double(iv) / grp_vals.number_of_elements();
                }
            }

            dom["fields"]["mapback_global_vids"].set(dom["fields/global_vertex_ids"]);
            dom["fields"]["mapback_braid"].set(dom["fields/braid"]);
            dom["fields"]["mapback_radial"].set(dom["fields/radial"]);
        }

        for (const Node* pdom : repart_mesh_doms)
        {
            const Node& dom = *pdom;
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

    Node mapback_opts;
    mapback_opts["fields"].append().set("mapback_braid");
    mapback_opts["fields"].append().set("mapback_radial");
    mapback_opts["fields"].append().set("mapback_global_vids");

    // Perform a map-back of some zone-centered variables
    conduit::blueprint::mpi::mesh::partition_map_back(repart_mesh,
                                                      mapback_opts,
                                                      mesh,
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
        opts["field_names"].append().set("global_vertex_ids");
        opts["field_names"].append().set("mapback_global_vids");
        opts["field_names"].append().set("parmetis_result");
        opts["field_names"].append().set("braid");
        opts["field_names"].append().set("radial");
        opts["field_names"].append().set("mapback_braid");
        opts["field_names"].append().set("mapback_radial");

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
TEST(blueprint_mpi_parmetis, empty_mesh)
{
    Node mesh;
    Node part_opts;
    part_opts["partition"] = 3;
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,
                                                            part_opts,
                                                            MPI_COMM_WORLD);
}


//-----------------------------------------------------------------------------
TEST(blueprint_mpi_parmetis, empty_mesh_on_non_root_rank)
{
    //
    // note: parmetis will give up in this case
    //

    int par_size, par_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &par_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &par_rank);
    
    Node mesh;
    if(par_rank == 0)
    {
        conduit::blueprint::mesh::examples::braid("quads",10,10,0,mesh);
    }
    Node part_opts;
    part_opts["partition"] = 3;
    conduit::blueprint::mpi::mesh::generate_partition_field(mesh,
                                                            part_opts,
                                                            MPI_COMM_WORLD);
                                                            

    std::string output_base = "tout_bp_mpi_mesh_parametis_empty_mesh_on_non_root_rank";

    conduit::relay::mpi::io::blueprint::save_mesh(mesh,
            output_base,
            "hdf5",
            // opts,
            MPI_COMM_WORLD);
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
