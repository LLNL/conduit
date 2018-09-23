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
static const index_t ELEM_TYPE_LINES[4]         = {     1,       0,     -1,      0};
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

index_t calc_mesh_lines(index_t type, const index_t *npts, bool unique = true)
{
    index_t num_elems = calc_mesh_elems(type, npts, true);

    index_t num_lines = 0, num_int_lines = 0;
    for(index_t di = 0; di < ELEM_TYPE_DIMS[type]; di++)
    {
        index_t dim_num_lines = npts[di] - 1;
        index_t dim_num_int_lines = npts[di] - 1;
        for(index_t dj = 0; dj < ELEM_TYPE_DIMS[type]; dj++)
        {
            dim_num_lines *= (di != dj) ? npts[dj] : 1;
            dim_num_int_lines *= (di != dj) ? npts[dj] - 2 : 1;
        }
        num_lines += dim_num_lines;
        num_int_lines += dim_num_int_lines;
    }

    return (num_lines + (ELEM_TYPE_LINES[type] * num_elems) + !unique * (
        num_int_lines + (ELEM_TYPE_LINES[type] * num_elems)));
}

index_t calc_mesh_faces(index_t type, const index_t *npts, bool unique = true)
{
    index_t num_faces = 0, num_int_faces = 0;

    for(index_t di = 0; di < ELEM_TYPE_DIMS[type]; di++)
    {
        index_t dim_num_faces = npts[di];
        index_t dim_num_int_faces = npts[di] - 2;
        for(index_t dj = 0; dj < ELEM_TYPE_DIMS[type]; dj++)
        {
            dim_num_faces *= (di != dj) ? npts[dj] - 1 : 1;
            dim_num_int_faces *= (di != dj) ? npts[dj] - 1 : 1;
        }
        num_faces += dim_num_faces;
        num_int_faces += dim_num_int_faces;
    }

    return num_faces + !unique * num_int_faces;
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
            EXPECT_EQ(nonpoly_offset_data[oi], oi * elem_indices);
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
            EXPECT_EQ(poly_offset_data[oi],
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
            // NOTE: The following lines are for debugging purposes only.
            std::string topo_type((mi == 0) ? "implicit" : "explicit");
            std::cout << "  " << topo_type << " source topology..." << std::endl;

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

                Node &mesh_axis = coords["values"][coord_axis];
                Node &cent_axis = cent_coords["values"][coord_axis];

                EXPECT_EQ(cent_axis.dtype().id(), mesh_axis.dtype().id());
                EXPECT_EQ(cent_axis.dtype().number_of_elements(), mesh_elems);
            }

            // Verify Correctness of Topology //

            Node &mesh_conn = mesh_topo["elements/connectivity"];
            Node &cent_conn = cent_topo["elements/connectivity"];

            EXPECT_EQ(cent_topo["coordset"].as_string(), CENTROID_COORDSET_NAME);
            EXPECT_EQ(cent_conn.dtype().id(), mesh_conn.dtype().id());
            EXPECT_EQ(cent_conn.dtype().number_of_elements(), mesh_elems);

            // TODO(JRC): Extend this test case to validate that each centroid is
            // contained within the convex hull of its source element.
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_lines)
{
    const index_t MPDIMS[3] = {4, 4, 4};

    const std::string LINE_TOPOLOGY_NAME = "ltopo";

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const index_t mesh_lines = calc_mesh_lines(ti, &MPDIMS[0], true);
        const bool is_mesh_3d = ELEM_TYPE_DIMS[ti] == 3;

        // NOTE: Skip values indicated to have an invalid subline scheme.
        const bool is_mesh_lineworthy = ELEM_TYPE_LINES[ti] != -1;
        if(!is_mesh_lineworthy) { continue; }

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing line mesh generation for type '" <<
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

            Node line_mesh;
            Node &line_coords = line_mesh["coordsets"][coords.name()];
            line_coords.set_external(coords);

            Node &line_topo = line_mesh["topologies"][LINE_TOPOLOGY_NAME];
            mesh::topology::unstructured::generate_lines(mesh_topo, line_topo);

            // General Data/Schema Checks //

            EXPECT_TRUE(mesh::topology::unstructured::verify(line_topo, info));

            EXPECT_EQ(line_topo["coordset"].as_string(), coords.name());
            EXPECT_EQ(line_topo["elements/shape"].as_string(), "line");

            Node &line_conn = line_topo["elements/connectivity"];
            EXPECT_EQ(line_conn.dtype().id(), mesh_conn.dtype().id());
            EXPECT_EQ(line_conn.dtype().number_of_elements() / 2, mesh_lines);

            // Content Consistency Checks //

            // TODO(JRC): Extend this test case so that it more thoroughly checks
            // the contents of the unique line mesh.
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_sides)
{
    const index_t MPDIMS[3] = {4, 4, 4};

    const std::string SIDE_COORDSET_NAME = "scoords";
    const std::string SIDE_TOPOLOGY_NAME = "stopo";
    const std::string SIDE_FIELD_NAME = "sfield";

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const bool is_mesh_3d = ELEM_TYPE_DIMS[ti] == 3;
        const index_t mesh_elems = calc_mesh_elems(ti, &MPDIMS[0]);
        const index_t mesh_faces = calc_mesh_faces(ti, &MPDIMS[0], true);

        const index_t si = is_mesh_3d ? 2 : 0;
        const std::string elem_side_type =
            ELEM_TYPE_LIST[si].substr(0, ELEM_TYPE_LIST[si].size() - 1);
        const index_t sides_per_elem =
            ELEM_TYPE_FACES[ti] * ELEM_TYPE_FACE_INDICES[ti];
        const index_t mesh_sides = mesh_elems * sides_per_elem;

        // NOTE: Skip values indicated to have an invalid subline scheme.
        const bool is_mesh_lineworthy = ELEM_TYPE_LINES[ti] != -1;
        if(!is_mesh_lineworthy) { continue; }

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing side generation for type '" <<
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

            Node side_mesh;
            Node &side_coords = side_mesh["coordsets"][SIDE_COORDSET_NAME];
            Node &side_topo = side_mesh["topologies"][SIDE_TOPOLOGY_NAME];
            Node &side_field = side_mesh["fields"][SIDE_FIELD_NAME];
            mesh::topology::unstructured::generate_sides(
                mesh_topo, side_topo, side_coords, side_field);

            EXPECT_TRUE(mesh::coordset::_explicit::verify(side_coords, info));
            EXPECT_TRUE(mesh::topology::unstructured::verify(side_topo, info));
            EXPECT_TRUE(mesh::field::verify(side_field, info));

            // Verify Correctness of Coordset //

            const std::vector<std::string> coord_axes = coords["values"].child_names();
            for(index_t ci = 0; ci < (index_t)coord_axes.size(); ci++)
            {
                const std::string &coord_axis = coord_axes[ci];
                EXPECT_TRUE(side_coords["values"].has_child(coord_axis));

                Node &mesh_axis = coords["values"][coord_axis];
                Node &side_axis = side_coords["values"][coord_axis];

                EXPECT_EQ(side_axis.dtype().id(), mesh_axis.dtype().id());
                EXPECT_EQ(side_axis.dtype().number_of_elements(),
                    mesh_axis.dtype().number_of_elements() + mesh_elems +
                    is_mesh_3d  * mesh_faces);
            }

            // Verify Correctness of Topology //

            Node &mesh_conn = mesh_topo["elements/connectivity"];
            Node &side_conn = side_topo["elements/connectivity"];

            Node &side_off = side_topo["elements/offsets"];
            mesh::topology::unstructured::generate_offsets(side_topo, side_off);

            EXPECT_EQ(side_topo["coordset"].as_string(), SIDE_COORDSET_NAME);
            EXPECT_EQ(side_topo["elements/shape"].as_string(), elem_side_type);
            EXPECT_EQ(side_conn.dtype().id(), mesh_conn.dtype().id());
            EXPECT_EQ(side_off.dtype().number_of_elements(), mesh_sides);

            // TODO(JRC): Augment this test case to verify that all of the given
            // elements have the expected area/volume.

            // Verify Correctness of Map Field //

            EXPECT_EQ(side_field["association"].as_string(), "element");
            EXPECT_EQ(side_field["values"].dtype().number_of_elements(), mesh_sides);

            { // Validate Contents of Map Field //
                Node side_field_int64;
                side_field["values"].to_int64_array(side_field_int64);
                int64_array side_field_data = side_field_int64.as_int64_array();

                std::vector<int64> side_expected_array(mesh_sides);
                for(index_t si = 0; si < side_expected_array.size();)
                {
                    for(index_t esi = 0; esi < sides_per_elem; si++, esi++)
                    {
                        side_expected_array[si] = si / sides_per_elem;
                    }
                }
                int64_array side_expected_data(&side_expected_array[0],
                    DataType::int64(mesh_sides));

                EXPECT_FALSE(side_field_data.diff(side_expected_data, info));
            }
        }
    }
}

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_generate_unstructured, generate_corners)
{
    const index_t MPDIMS[3] = {4, 4, 4};

    const std::string CORNER_COORDSET_NAME = "ccoords";
    const std::string CORNER_TOPOLOGY_NAME = "ctopo";
    const std::string CORNER_FIELD_NAME = "cfield";

    for(index_t ti = 0; ti < 4; ti++)
    {
        const std::string &elem_type = ELEM_TYPE_LIST[ti];
        const bool is_mesh_3d = ELEM_TYPE_DIMS[ti] == 3;
        const index_t mesh_lines = calc_mesh_lines(ti, &MPDIMS[0], true);
        const index_t mesh_faces = calc_mesh_faces(ti, &MPDIMS[0], true);
        const index_t mesh_elems = calc_mesh_elems(ti, &MPDIMS[0]);

        const std::string elem_corner_type = is_mesh_3d ?
            "polyhedral" : "polygonal";
        const index_t corners_per_elem = ELEM_TYPE_INDICES[ti];
        const index_t mesh_corners = corners_per_elem * mesh_elems;

        // NOTE: Skip values indicated to have an invalid subline scheme.
        const bool is_mesh_lineworthy = ELEM_TYPE_LINES[ti] != -1;
        if(!is_mesh_lineworthy) { continue; }

        // NOTE: The following lines are for debugging purposes only.
        std::cout << "Testing corner generation for type '" <<
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

            Node corner_mesh;
            Node &corner_coords = corner_mesh["coordsets"][CORNER_COORDSET_NAME];
            Node &corner_topo = corner_mesh["topologies"][CORNER_TOPOLOGY_NAME];
            Node &corner_field = corner_mesh["fields"][CORNER_FIELD_NAME];
            mesh::topology::unstructured::generate_corners(
                mesh_topo, corner_topo, corner_coords, corner_field);

            // Verify Correctness of Coordset //

            const std::vector<std::string> coord_axes = coords["values"].child_names();
            for(index_t ci = 0; ci < (index_t)coord_axes.size(); ci++)
            {
                const std::string &coord_axis = coord_axes[ci];
                EXPECT_TRUE(corner_coords["values"].has_child(coord_axis));

                Node &mesh_axis = coords["values"][coord_axis];
                Node &corner_axis = corner_coords["values"][coord_axis];

                EXPECT_EQ(corner_axis.dtype().id(), mesh_axis.dtype().id());
                EXPECT_EQ(corner_axis.dtype().number_of_elements(),
                    mesh_axis.dtype().number_of_elements() + mesh_lines +
                    is_mesh_3d * mesh_faces + mesh_elems);
            }

            // // Verify Correctness of Topology //

            Node &mesh_conn = mesh_topo["elements/connectivity"];
            Node &corner_conn = corner_topo["elements/connectivity"];

            Node &corner_off = corner_topo["elements/offsets"];
            mesh::topology::unstructured::generate_offsets(corner_topo, corner_off);

            EXPECT_EQ(corner_topo["coordset"].as_string(), CORNER_COORDSET_NAME);
            EXPECT_EQ(corner_topo["elements/shape"].as_string(), elem_corner_type);
            EXPECT_EQ(corner_conn.dtype().id(), mesh_conn.dtype().id());
            EXPECT_EQ(corner_off.dtype().number_of_elements(), mesh_corners);

            // Verify Correctness of Map Field //

            EXPECT_EQ(corner_field["association"].as_string(), "element");
            EXPECT_EQ(corner_field["values"].dtype().number_of_elements(), mesh_corners);

            { // Validate Contents of Map Field //
                Node corner_field_int64;
                corner_field["values"].to_int64_array(corner_field_int64);
                int64_array corner_field_data = corner_field_int64.as_int64_array();

                std::vector<int64> corner_expected_array(mesh_corners);
                for(index_t si = 0; si < corner_expected_array.size();)
                {
                    for(index_t ssi = 0; ssi < corners_per_elem; si++, ssi++)
                    {
                        corner_expected_array[si] = si / corners_per_elem;
                    }
                }
                int64_array corner_expected_data(&corner_expected_array[0],
                    DataType::int64(corner_expected_array.size()));

                EXPECT_FALSE(corner_field_data.diff(corner_expected_data, info));
            }
        }
    }
}
