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
/// file: t_blueprint_mesh_generate.cpp
///
//-----------------------------------------------------------------------------

#if defined(CONDUIT_PLATFORM_WINDOWS)
#define NOMINMAX
#undef min
#undef max
#include "Windows.h"
#endif

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::blueprint;
using namespace conduit::utils;

/// Testing Constants ///

static const std::string ELEM_TYPE_LIST[4]      = {"tris", "quads", "tets", "hexs"};
static const index_t ELEM_TYPE_DIMS[4]          = {     2,       2,      3,      3};
static const index_t ELEM_TYPE_SUBELEMS[4]      = {     2,       1,      6,      1};
static const index_t ELEM_TYPE_INDICES[4]       = {     3,       4,      4,      8};
static const index_t ELEM_TYPE_EDGES[4]         = {     1,       0,     -1,      0};
static const index_t ELEM_TYPE_FACES[4]         = {     1,       1,      4,      6};
static const index_t ELEM_TYPE_FACE_INDICES[4]  = {     3,       4,      3,      4};

/// Testing Helpers ///

index_t calc_mesh_elems(index_t type, const index_t *npts, bool super = false)
{
    index_t num_elems = 1;

    for(index_t di = 0; di < ELEM_TYPE_DIMS[type]; di++)
    {
        num_elems *= (npts[di] - 1);
    }

    return (super ? 1 : ELEM_TYPE_SUBELEMS[type]) * num_elems;
}

index_t calc_mesh_edges(index_t type, const index_t *npts, bool unique)
{
    index_t num_elems = calc_mesh_elems(type, npts, true);

    index_t num_edges = 0, num_int_edges = 0;
    for(index_t di = 0; di < ELEM_TYPE_DIMS[type]; di++)
    {
        index_t dim_num_edges = npts[di] - 1;
        index_t dim_num_int_edges = npts[di] - 1;
        for(index_t dj = 0; dj < ELEM_TYPE_DIMS[type]; dj++)
        {
            dim_num_edges *= (di != dj) ? npts[dj] : 1;
            dim_num_int_edges *= (di != dj) ? npts[dj] - 2 : 1;
        }
        num_edges += dim_num_edges;
        num_int_edges += dim_num_int_edges;
    }

    return num_edges + (ELEM_TYPE_EDGES[type] * num_elems) + !unique * (
        num_int_edges + (ELEM_TYPE_EDGES[type] * num_elems));
}

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_offsets_nonpoly)
{
    const index_t MPDIMS[3] = {3, 3, 3};

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_indices = ELEM_TYPE_INDICES[ti];
        const index_t mesh_elems = calc_mesh_elems(ti, &MPDIMS[0]);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing offset generation for nonpolygonal type '" <<
            elem_type << "'..." << std::endl;

        Node nonpoly_node;
        mesh::examples::braid(elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],nonpoly_node);
        Node &nonpoly_topo = nonpoly_node["topologies"].child(0);
        Node &nonpoly_conn = nonpoly_topo["elements/connectivity"];

        Node nonpoly_offsets;
        mesh::topology::unstructured::generate_offsets(nonpoly_topo, nonpoly_offsets);

        EXPECT_EQ(nonpoly_offsets.dtype().id(), nonpoly_conn.dtype().id());
        EXPECT_EQ(nonpoly_offsets.dtype().number_of_elements(), mesh_elems);

        Node nonpoly_offsets_int64;
        nonpoly_offsets.to_int64_array(nonpoly_offsets_int64);
        int64_array nonpoly_offset_data = nonpoly_offsets_int64.as_int64_array();
        for(index_t oi = 0; oi < nonpoly_offsets.dtype().number_of_elements(); oi++)
        {
            ASSERT_EQ(nonpoly_offset_data[oi], oi * elem_indices);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_offsets_poly)
{
    const index_t MPDIMS[3] = {3, 3, 3};

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_faces = ELEM_TYPE_FACES[ti];
        const index_t &elem_face_indices = ELEM_TYPE_FACE_INDICES[ti];
        const index_t mesh_elems = calc_mesh_elems(ti, &MPDIMS[0]);
        const bool is_elem_3d = ELEM_TYPE_DIMS[ti] == 3;

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing offset generation for polygonal type '" <<
            elem_type << "'..." << std::endl;

        Node nonpoly_node;
        mesh::examples::braid(elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],nonpoly_node);
        Node &nonpoly_topo = nonpoly_node["topologies"].child(0);

        Node poly_topo, poly_offsets;
        mesh::topology::unstructured::to_polygonal(nonpoly_topo, poly_topo);
        mesh::topology::unstructured::generate_offsets(poly_topo, poly_offsets);
        Node &poly_conn = poly_topo["elements/connectivity"];

        EXPECT_EQ(poly_offsets.dtype().id(), poly_conn.dtype().id());
        EXPECT_EQ(poly_offsets.dtype().number_of_elements(), mesh_elems);

        Node poly_offsets_int64;
        poly_offsets.to_int64_array(poly_offsets_int64);
        int64_array poly_offset_data = poly_offsets_int64.as_int64_array();
        for(index_t oi = 0; oi < poly_offsets.dtype().number_of_elements(); oi++)
        {
            ASSERT_EQ(poly_offset_data[oi],
                oi * (is_elem_3d + elem_faces * (1 + elem_face_indices)));
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_centroids)
{
    const index_t MPDIMS[3] = {3, 3, 3};

    const std::string CENTROID_COORDSET_NAME = "ccoords";
    const std::string CENTROID_TOPOLOGY_NAME = "ctopo";

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t mesh_elems = calc_mesh_elems(ti, &MPDIMS[0]);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing centroid generation for type '" <<
            elem_type << "'..." << std::endl;

        Node mesh, info;
        mesh::examples::braid(elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],mesh);
        Node &coords = mesh["coordsets"].child(0);
        Node &topo = mesh["topologies"].child(0);

        Node &poly_topo = mesh["topologies"]["poly_" + topo.name()];
        mesh::topology::unstructured::to_polygonal(topo, poly_topo);

        Node *mesh_topos[] = {&topo, &poly_topo};
        for(index_t mi = 0; mi < 2; mi++)
        {
            Node &mesh_topo = *mesh_topos[mi];

            Node cent_mesh;
            Node &cent_coords = cent_mesh["coordsets"][CENTROID_COORDSET_NAME];
            Node &cent_topo = cent_mesh["topologies"][CENTROID_TOPOLOGY_NAME];
            mesh::topology::unstructured::generate_centroids(
                mesh_topo, cent_topo, cent_coords);

            EXPECT_TRUE(mesh::coordset::_explicit::verify(cent_coords, info));
            EXPECT_TRUE(mesh::topology::unstructured::verify(cent_topo, info));

            // Verify Correctness of Coordset //

            const std::vector<std::string> coord_axes = coords["values"].child_names();
            for(index_t ci = 0; ci < (index_t)coord_axes.size(); ci++)
            {
                const std::string &coord_axis = coord_axes[ci];
                EXPECT_TRUE(cent_coords["values"].has_child(coord_axis));

                EXPECT_EQ(cent_coords["values"][coord_axis].dtype().id(),
                    coords["values"][coord_axis].dtype().id());
                EXPECT_EQ(cent_coords["values"][coord_axis].dtype().number_of_elements(),
                    mesh_elems);
            }

            // Verify Correctness of Topology //

            EXPECT_EQ(cent_topo["coordset"].as_string(), CENTROID_COORDSET_NAME);
            EXPECT_EQ(cent_topo["elements/connectivity"].dtype().id(),
                mesh_topo["elements/connectivity"].dtype().id());
            EXPECT_EQ(cent_topo["elements/connectivity"].dtype().number_of_elements(),
                mesh_elems);

            // TODO(JRC): Extend this test case to validate that each centroid is
            // contained within the convex hull of its source element.
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_edges)
{
    const index_t MPDIMS[3] = {2, 2, 2};

    const std::string UNIQUE_TOPOLOGY_NAME = "etopo_unique";
    const std::string TOTAL_TOPOLOGY_NAME = "etopo_total";

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t mesh_unique_edges = calc_mesh_edges(ti, &MPDIMS[0], true);
        const index_t mesh_total_edges = calc_mesh_edges(ti, &MPDIMS[0], false);
        const bool is_mesh_3d = ELEM_TYPE_DIMS[ti] == 3;
        const bool is_mesh_edgeworthy = ELEM_TYPE_EDGES[ti] != -1;

        // NOTE: Skip values indicated to have an invalid subedge scheme.
        if(!is_mesh_edgeworthy) { continue; }

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing edge mesh generation for type '" <<
            elem_type << "'..." << std::endl;

        Node mesh, info;
        mesh::examples::braid(elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],mesh);
        Node &coords = mesh["coordsets"].child(0);
        Node &topo = mesh["topologies"].child(0);

        Node &poly_topo = mesh["topologies"]["poly_" + topo.name()];
        mesh::topology::unstructured::to_polygonal(topo, poly_topo);

        Node *mesh_topos[] = {&topo, &poly_topo};
        for(index_t mi = 0; mi < 2; mi++)
        {
            // NOTE: The following lines are for debugging purposes only.
            std::string topo_type((mi == 0) ? "implicit" : "explicit");
            std::cout << "  " << topo_type << " source topology..." << std::endl;

            Node &mesh_topo = *mesh_topos[mi];
            Node &mesh_conn = mesh_topo["elements/connectivity"];

            Node edge_mesh;
            Node &edge_coords = edge_mesh["coordsets"][coords.name()];
            edge_coords.set_external(coords);

            Node &unique_edge_topo = edge_mesh["topologies"][UNIQUE_TOPOLOGY_NAME];
            mesh::topology::unstructured::generate_edges(
                mesh_topo, true, unique_edge_topo);

            Node &total_edge_topo = edge_mesh["topologies"][TOTAL_TOPOLOGY_NAME];
            mesh::topology::unstructured::generate_edges(
                mesh_topo, false, total_edge_topo);

            // General Data/Schema Checks //

            Node *edge_topos[] = {&unique_edge_topo, &total_edge_topo};
            index_t edge_topo_lengths[] = {mesh_unique_edges, mesh_total_edges};
            for(index_t emi = 0; emi < 2; emi++)
            {
                // NOTE: Skip 3D mesh total edge checks because their closed forms
                // for edge counts are too complicated.
                if(is_mesh_3d && emi == 1) { continue; }

                // NOTE: The following lines are for debugging purposes only.
                std::string edge_type((emi == 0) ? "unique" : "non-unique");
                std::cout << "    " << edge_type << " edge topology..." << std::endl;

                Node &edge_topo = *edge_topos[emi];
                index_t topo_length = edge_topo_lengths[emi];
                EXPECT_TRUE(mesh::topology::unstructured::verify(edge_topo, info));

                EXPECT_EQ(edge_topo["coordset"].as_string(), coords.name());
                EXPECT_EQ(edge_topo["elements/shape"].as_string(), "line");

                Node &edge_conn = edge_topo["elements/connectivity"];
                EXPECT_EQ(edge_conn.dtype().id(), mesh_conn.dtype().id());
                EXPECT_EQ(edge_conn.dtype().number_of_elements() / 2, topo_length);
            }

            // Content Consistency Checks //

            // TODO(JRC): Extend this test case so that it more thoroughly checks
            // the contents of the unique edge mesh.

            std::vector< std::pair<index_t, index_t> > edge_lists[2];
            for(index_t emi = 0; emi < 2; emi++)
            {
                Node &edge_topo = *edge_topos[emi];
                Node &edge_conn = edge_topo["elements/connectivity"];
                std::vector< std::pair<index_t, index_t> > &edge_list = edge_lists[emi];

                Node edge_conn_data_node;
                edge_conn.to_int64_array(edge_conn_data_node);
                int64_array edge_conn_data = edge_conn_data_node.as_int64_array();
                for(index_t i = 0; i < edge_conn_data.number_of_elements(); i += 2)
                {
                    edge_list.push_back(std::pair<index_t, index_t>(
                        edge_conn_data[i+0], edge_conn_data[i+1]));
                }

                // NOTE: Required in order to perform inclusion test below.
                std::sort(edge_list.begin(), edge_list.end());
            }

            // total edge list is a superset of the unique edge list
            std::vector< std::pair<index_t, index_t> >
                &unique_edges = edge_lists[0], &total_edges = edge_lists[1];
            ASSERT_LE(unique_edges.size(), total_edges.size());
            ASSERT_TRUE(std::includes(
                total_edges.begin(), total_edges.end(),
                unique_edges.begin(), unique_edges.end()));

            // total edge set only contains items in unique edge set
            std::set< std::pair<index_t, index_t> > edge_sets[2];
            for(index_t emi = 0; emi < 2; emi++)
            {
                std::set< std::pair<index_t, index_t> > &edge_set = edge_sets[emi];
                for(index_t ei = 0; ei < (index_t)edge_lists[emi].size(); ei++)
                {
                    std::pair<index_t, index_t> curr_edge = edge_lists[emi][ei];
                    edge_set.insert(std::pair<index_t, index_t>(
                        std::min(curr_edge.first, curr_edge.second),
                        std::max(curr_edge.first, curr_edge.second)));
                }
            }

            std::set< std::pair<index_t, index_t> >
                &unique_edge_set = edge_sets[0], &total_edge_set = edge_sets[1];
            ASSERT_EQ(total_edge_set, unique_edge_set);
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_2d)
{
    // TODO(JRC): Implement this function.
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides_3d)
{
    // TODO(JRC): Implement this function.
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_corners_2d)
{
    // TODO(JRC): Implement this function.
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_corners_3d)
{
    // TODO(JRC): Implement this function.
}
