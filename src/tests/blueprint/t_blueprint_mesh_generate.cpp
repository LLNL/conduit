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

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_relay.hpp"
#include "conduit_log.hpp"

#include <cmath>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

/// Testing Constants ///

static const std::string ELEM_TYPE_LIST[4]      = {"tris", "quads", "tets", "hexs"};
static const index_t ELEM_TYPE_SUBELEMS[4]      = {     2,       1,      6,      1};
static const index_t ELEM_TYPE_INDICES[4]       = {     3,       4,      4,      8};
static const index_t ELEM_TYPE_FACES[4]         = {     1,       1,      4,      6};
static const index_t ELEM_TYPE_FACE_INDICES[4]  = {     3,       4,      3,      4};

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_offsets_nonpoly)
{
    const index_t MPDIMS[3] = {3, 3, 3};
    const index_t MEDIMS[3] = {MPDIMS[0]-1, MPDIMS[1]-1, MPDIMS[2]-1};

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_subelems = ELEM_TYPE_SUBELEMS[ti];
        const index_t &elem_indices = ELEM_TYPE_INDICES[ti];
        const bool is_elem_3d = ELEM_TYPE_FACES[ti] > 1;
        const index_t mesh_elems = elem_subelems *
            MEDIMS[0] * MEDIMS[1] * (is_elem_3d ? MEDIMS[2] : 1);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing offset generation for nonpolygonal type '" <<
            elem_type << "'..." << std::endl;

        Node nonpoly_node;
        blueprint::mesh::examples::braid(
            elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],nonpoly_node);
        Node &nonpoly_topo = nonpoly_node["topologies"].child(0);

        Node nonpoly_offsets;
        blueprint::mesh::topology::unstructured::generate_offsets(
            nonpoly_topo, nonpoly_offsets);

        EXPECT_EQ(nonpoly_offsets.dtype().id(),
            nonpoly_topo["elements/connectivity"].dtype().id());
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
    const index_t MEDIMS[3] = {MPDIMS[0]-1, MPDIMS[1]-1, MPDIMS[2]-1};

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_subelems = ELEM_TYPE_SUBELEMS[ti];
        const index_t &elem_faces = ELEM_TYPE_FACES[ti];
        const index_t &elem_face_indices = ELEM_TYPE_FACE_INDICES[ti];
        const bool is_elem_3d = ELEM_TYPE_FACES[ti] > 1;
        const index_t mesh_elems = elem_subelems *
            MEDIMS[0] * MEDIMS[1] * (is_elem_3d ? MEDIMS[2] : 1);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing offset generation for polygonal type '" <<
            elem_type << "'..." << std::endl;

        Node nonpoly_node;
        blueprint::mesh::examples::braid(
            elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],nonpoly_node);
        Node &nonpoly_topo = nonpoly_node["topologies"].child(0);

        Node poly_topo, poly_offsets;
        blueprint::mesh::topology::unstructured::to_polygonal(nonpoly_topo, poly_topo);
        blueprint::mesh::topology::unstructured::generate_offsets(poly_topo, poly_offsets);

        EXPECT_EQ(poly_offsets.dtype().id(),
            poly_topo["elements/connectivity"].dtype().id());
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
    const index_t MEDIMS[3] = {MPDIMS[0]-1, MPDIMS[1]-1, MPDIMS[2]-1};

    const std::string CENTROID_COORDSET_NAME = "ccoords";
    const std::string CENTROID_TOPOLOGY_NAME = "ctopo";

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_subelems = ELEM_TYPE_SUBELEMS[ti];
        const bool is_mesh_3d = ELEM_TYPE_FACES[ti] > 1;
        const index_t mesh_elems = elem_subelems *
            MEDIMS[0] * MEDIMS[1] * (is_mesh_3d ? MEDIMS[2] : 1);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing centroid generation for type '" <<
            elem_type << "'..." << std::endl;

        Node mesh, info;
        blueprint::mesh::examples::braid(elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],mesh);
        Node &coords = mesh["coordsets"].child(0);
        Node &topo = mesh["topologies"].child(0);

        Node &poly_topo = mesh["topologies"]["poly_" + topo.name()];
        blueprint::mesh::topology::unstructured::to_polygonal(topo, poly_topo);

        Node *mesh_topos[] = {&topo, &poly_topo};
        for(index_t mi = 0; mi < 2; mi++)
        {
            Node &mesh_topo = *mesh_topos[mi];

            Node cent_mesh;
            Node& cent_coords = cent_mesh["coordsets"][CENTROID_COORDSET_NAME];
            Node& cent_topo = cent_mesh["topologies"][CENTROID_TOPOLOGY_NAME];
            blueprint::mesh::topology::unstructured::generate_centroids(
                mesh_topo, cent_topo, cent_coords);

            EXPECT_TRUE(blueprint::mesh::coordset::_explicit::verify(cent_coords, info));
            EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(cent_topo, info));

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
TEST(conduit_blueprint_generate_unstructured, generate_edges_unique)
{
    // NOTE: This variable describes the number of internal edges contained
    // within each superelement of the base mesh. "tets" are given a salient
    // value in order to indicate the the overlap scheme is too complicated
    // and should be skipped.
    const index_t ELEM_TYPE_EDGES[4] = {1, 0, -1, 0};

    const index_t MPDIMS[3] = {4, 4, 4};
    const index_t MEDIMS[3] = {MPDIMS[0]-1, MPDIMS[1]-1, MPDIMS[2]-1};

    const std::string EDGE_TOPOLOGY_NAME = "etopo";

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t &elem_subedges = ELEM_TYPE_EDGES[ti];
        const bool is_mesh_3d = ELEM_TYPE_FACES[ti] > 1;

        // NOTE: Skip values indicated to have an invalid subedge scheme.
        if(elem_subedges < 0) { continue; }

        const index_t mesh_dims = is_mesh_3d ? 3 : 2;
        const index_t mesh_elems = MEDIMS[0] * MEDIMS[1] * (is_mesh_3d ? MEDIMS[2] : 1);

        index_t mesh_edges_acc = 0;
        for(index_t di = 0; di < mesh_dims; di++)
        {
            index_t dim_num_edges = MPDIMS[di] - 1;
            for(index_t dj = 0; dj < mesh_dims; dj++)
            {
                dim_num_edges *= (di != dj) ? MPDIMS[dj] : 1;
            }
            mesh_edges_acc += dim_num_edges;
        }
        const index_t mesh_edges = mesh_edges_acc + (mesh_elems * elem_subedges);

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing edge mesh generation for type '" <<
            elem_type << "'..." << std::endl;

        Node mesh, info;
        blueprint::mesh::examples::braid(elem_type,MPDIMS[0],MPDIMS[1],MPDIMS[2],mesh);
        Node &coords = mesh["coordsets"].child(0);
        Node &topo = mesh["topologies"].child(0);

        Node &poly_topo = mesh["topologies"]["poly_" + topo.name()];
        blueprint::mesh::topology::unstructured::to_polygonal(topo, poly_topo);

        Node *mesh_topos[] = {&topo, &poly_topo};
        for(index_t mi = 0; mi < 2; mi++)
        {
            Node &mesh_topo = *mesh_topos[mi];

            Node edge_mesh;
            Node& edge_coords = edge_mesh["coordsets"][coords.name()];
            Node& edge_topo = edge_mesh["topologies"][EDGE_TOPOLOGY_NAME];
            edge_coords.set_external(coords);
            blueprint::mesh::topology::unstructured::generate_edges(
                mesh_topo, true, edge_topo);

            EXPECT_TRUE(blueprint::mesh::topology::unstructured::verify(edge_topo, info));

            EXPECT_EQ(edge_topo["coordset"].as_string(), coords.name());
            EXPECT_EQ(edge_topo["elements/shape"].as_string(), "line");

            EXPECT_EQ(edge_topo["elements/connectivity"].dtype().id(),
                mesh_topo["elements/connectivity"].dtype().id());
            EXPECT_EQ(edge_topo["elements/connectivity"].dtype().number_of_elements() / 2,
                mesh_edges);

            // TODO(JRC): Extend this test case to do further validation based
            // on the individual edges in the source geometry.
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_edges_all)
{
    // TODO(JRC): Implement this function.
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
