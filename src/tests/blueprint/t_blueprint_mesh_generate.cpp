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

float64 calc_mesh_elem_volume(index_t type, const index_t *npts)
{
    // NOTE(JRC): This is explicitly given in the definition of the 'braid'
    // example generation function.
    const float64 dim_length = 20.0;

    float64 elem_volume = 1.0;
    for(index_t di = 0; di < ELEM_TYPE_DIMS[type]; di++)
    {
        elem_volume *= dim_length / (npts[di] - 1.0);
    }
    elem_volume /= ELEM_TYPE_SUBELEMS[type];

    return elem_volume;
}

// TODO(JRC): The fact that there isn't a standard C++ library for simple
// linear algebra operations and that this is the ~20th time I've had to
// write such operations makes me sad indeed.

bool fuzzy_eq(float64 f1, float64 f2, float64 epsilon = CONDUIT_EPSILON)
{
    return std::abs(f1 - f2) <= epsilon;
}

bool fuzzy_le(float64 f1, float64 f2, float64 epsilon = CONDUIT_EPSILON)
{
    return f1 < f2 || fuzzy_eq(f1, f2, epsilon);
}

void calc_vec_add(const float64 *u, const float64* v, float64 *r)
{
    r[0] = u[0] + v[0];
    r[1] = u[1] + v[1];
    r[2] = u[2] + v[2];
}

void calc_vec_sub(const float64 *u, const float64* v, float64 *r)
{
    r[0] = u[0] - v[0];
    r[1] = u[1] - v[1];
    r[2] = u[2] - v[2];
}

void calc_vec_cross(const float64 *u, const float64 *v, float64 *r)
{
    r[0] = u[1] * v[2] - u[2] * v[1];
    r[1] = u[2] * v[0] - u[0] * v[2];
    r[2] = u[0] * v[1] - u[1] * v[0];
}

float64 calc_vec_mag(const float64 *u)
{
    return std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
}

void calc_inversion_field(index_t type, const Node &topo, const Node &coords, Node &dest)
{
    // NOTE(JRC): This function assumes the existence of an offsets field in
    // the source topology, which is true for all callees in this test suite.
    // TODO(JRC): The performance of this method would be greatly enhanced if a
    // sparse data table was given by "generate_lines" that indicated the relations
    // between source entities and the resulting lines (and vice versa).
    // TODO(JRC): If the type is 3D, then this field isn't absolutely correct about
    // the number of inversions because line-plane intersections aren't calculated.
    const index_t elem_npts[] = {2, 2, 2};
    const std::string elem_axes[] = {"x", "y", "z"};

    const Node &topo_conn = topo["elements/connectivity"];
    const Node &topo_off = topo["elements/offsets"];
    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType off_dtype(topo_off.dtype().id(), 1);
    const index_t topo_num_elems = topo_off.dtype().number_of_elements();

    Node data_node;

    std::vector< std::vector< std::set<index_t> > > elem_lines(topo_num_elems);
    {
        Node elem_mesh;
        Node &elem_coords = elem_mesh["coordsets"][coords.name()];
        Node &elem_topo = elem_mesh["topologies"][coords.name()];
        elem_coords.set_external(coords);

        Node line_mesh;
        Node &line_coords = line_mesh["coordsets"][coords.name()];
        Node &line_topo = line_mesh["topologies"][topo.name()];
        line_coords.set_external(coords);

        Node elem_topo_templ;
        elem_topo_templ.set_external(topo);
        elem_topo_templ.remove("elements/connectivity");
        elem_topo_templ.remove("elements/offsets");

        int64 line_data_raw[2] = {-1, -1};
        Node line_data(DataType::int64(2), &line_data_raw[0], true);

        elem_topo.set(elem_topo_templ);
        Node &elem_conn = elem_topo["elements/connectivity"];
        for(index_t ei = 0; ei < topo_num_elems; ei++)
        {
            // TODO(JRC): This code was lifted directly from the private structure
            // 'conduit_blueprint_mesh.cpp:TopologyMetadata'. Ultimately, it would
            // be better if these two pieces of functionality were integrated.
            data_node.set_external(off_dtype, (void*)topo_off.element_ptr(ei));
            index_t elem_start_index = data_node.to_int64();
            data_node.set_external(off_dtype, (void*)topo_off.element_ptr(ei+1));
            index_t elem_end_index = (ei < topo_num_elems - 1) ?
                data_node.to_int64() : topo_conn.dtype().number_of_elements();

            index_t elem_size = elem_end_index - elem_start_index;
            data_node.set_external(DataType(conn_dtype.id(), elem_size),
                (void*)topo_conn.element_ptr(elem_start_index));
            data_node.to_data_type(DataType::int64(1).id(), elem_conn);

            mesh::topology::unstructured::generate_lines(elem_topo, line_topo);

            Node &line_conn = line_topo["elements/connectivity"];
            for(index_t li = 0; li < line_conn.dtype().number_of_elements(); li += 2)
            {
                data_node.set_external(DataType(line_conn.dtype().id(), 2),
                    (void*)line_conn.element_ptr(li));
                data_node.to_data_type(line_data.dtype().id(), line_data);

                std::set<index_t> curr_line;
                curr_line.insert((index_t)line_data_raw[0]);
                curr_line.insert((index_t)line_data_raw[1]);
                elem_lines[ei].push_back(curr_line);
            }
        }
    }

    dest.reset();
    dest["association"].set("element");
    dest["volume_dependent"].set("false");
    dest["topology"].set(topo.name());
    dest["values"].set(DataType::int32(topo_num_elems));

    int32_array dest_vals = dest["values"].as_int32_array();
    for(index_t ei = 0; ei < topo_num_elems; ei++)
    {
        const std::set< std::set<index_t> > elem_lineset(
            elem_lines[ei].begin(), elem_lines[ei].end());

        dest_vals[ei] = 0 - (elem_lines[ei].size() != elem_lineset.size());
        for(std::set< std::set<index_t> >::iterator iline_it = elem_lineset.begin();
            iline_it != elem_lineset.end() && dest_vals[ei] == 0; ++iline_it)
        {
            const std::set<index_t> &iline = *iline_it;
            for(std::set< std::set<index_t> >::iterator jline_it = elem_lineset.begin();
                jline_it != elem_lineset.end() && dest_vals[ei] == 0; ++jline_it)
            {
                const std::set<index_t> &jline = *jline_it;

                std::vector<index_t> ij_shared_points(2);
                std::vector<index_t>::iterator ij_shared_end = std::set_intersection(
                    iline.begin(), iline.end(), jline.begin(), jline.end(),
                    ij_shared_points.begin());

                // If there are no shared endpoints between the two input edges, then
                // we test for intersections.
                if(ij_shared_points.begin() == ij_shared_end)
                {
                    // Extract Coordinate Data //
                    index_t line_indices[2][2] = {{-1, -1}, {-1, -1}};
                    float64 line_starts[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
                    float64 line_ends[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
                    float64 line_vecs[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
                    for(index_t li = 0; li < 2; li++)
                    {
                        const std::set<index_t> &curr_line = (li == 0) ? iline : jline;
                        index_t *curr_indices = &line_indices[li][0];
                        float64 *curr_start = &line_starts[li][0];
                        float64 *curr_end = &line_ends[li][0];
                        float64 *curr_vec = &line_vecs[li][0];

                        curr_indices[0] = *(curr_line.begin());
                        curr_indices[1] = *(++curr_line.begin());

                        for(index_t di = 0; di < ELEM_TYPE_DIMS[type]; di++)
                        {
                            const Node &axis_coords = coords["values"][elem_axes[di]];
                            const DataType axis_dtype(axis_coords.dtype().id(), 1);

                            Node axis_data;
                            for(index_t ei = 0; ei < 2; ei++)
                            {
                                float64 *curr_point = (ei == 0) ? curr_start : curr_end;
                                axis_data.set_external(axis_dtype,
                                    (void*)axis_coords.element_ptr(curr_indices[ei]));
                                curr_point[di] = axis_data.to_float64();
                            }

                            curr_vec[di] = curr_end[di] - curr_start[di];
                        }
                    }

                    // Calculate Line Intersection //
                    float64 i2j_vec[3] = {0.0, 0.0, 0.0};
                    calc_vec_sub(&line_starts[1][0], &line_starts[0][0], &i2j_vec[0]);

                    float64 ixj_vec[3] = {0.0, 0.0, 0.0};
                    calc_vec_cross(&line_vecs[0][0], &line_vecs[1][0], &ixj_vec[0]);
                    float64 ixj_mag = calc_vec_mag(&i2j_vec[0]);
                    float64 i2jxi_vec[3] = {0.0, 0.0, 0.0};
                    calc_vec_cross(&i2j_vec[0], &line_vecs[0][0], &i2jxi_vec[0]);
                    float64 i2jxi_mag = calc_vec_mag(&i2jxi_vec[0]);
                    float64 i2jxj_vec[3] = {0.0, 0.0, 0.0};
                    calc_vec_cross(&i2j_vec[0], &line_vecs[1][0], &i2jxj_vec[0]);
                    float64 i2jxj_mag = calc_vec_mag(&i2jxj_vec[0]);

                    // NOTE: Evaluations based on algebraic derivations here:
                    // https://stackoverflow.com/a/565282
                    if(fuzzy_eq(ixj_mag, 0.0)) // parallel case
                    {
                        // colinear if true; parallel adjoint if false
                        dest_vals[ei] += fuzzy_eq(i2jxi_mag, 0.0);
                    }
                    else // non-parallel case
                    {
                        float64 iparam = i2jxj_mag / ixj_mag;
                        float64 jparam = i2jxi_mag / ixj_mag;

                        // intersect if line-line intersection exists in line
                        // segment parameter space of [0.0, 1.0] for both lines
                        dest_vals[ei] +=
                            (fuzzy_le(0.0, iparam) && fuzzy_le(iparam, 1.0)) &&
                            (fuzzy_le(0.0, jparam) && fuzzy_le(jparam, 1.0));
                    }
                }
            }
        }
    }
}

void calc_volume_field(index_t type, const Node &topo, const Node &coords, Node &dest)
{
    // NOTE(JRC): This function assumes the existence of an offsets field in
    // the source topology, which is true for all callees in this test suite.
    // TODO(JRC): This currently is only capable of calculating the hypervolume
    // of 2D topologies.
    const std::string elem_axes[] = {"x", "y", "z"};

    const Node &topo_conn = topo["elements/connectivity"];
    const Node &topo_off = topo["elements/offsets"];
    const DataType conn_dtype(topo_conn.dtype().id(), 1);
    const DataType off_dtype(topo_off.dtype().id(), 1);
    const bool topo_is_poly = topo["elements/shape"].as_string() == "polygonal";
    const index_t topo_num_elems = topo_off.dtype().number_of_elements();

    dest.reset();
    dest["association"].set("element");
    dest["volume_dependent"].set("false");
    dest["topology"].set(topo.name());
    dest["values"].set(DataType::float64(topo_num_elems));

    Node data_node;

    float64_array dest_vals = dest["values"].as_float64_array();
    for(index_t ei = 0; ei < topo_num_elems; ei++)
    {
        data_node.set_external(off_dtype, (void*)topo_off.element_ptr(ei));
        index_t elem_start_index = data_node.to_int64() + topo_is_poly;
        data_node.set_external(off_dtype, (void*)topo_off.element_ptr(ei+1));
        index_t elem_end_index = (ei < topo_num_elems - 1) ?
            data_node.to_int64() : topo_conn.dtype().number_of_elements();
        index_t elem_size = elem_end_index - elem_start_index;

        // NOTE(JRC): The polygonal volume calculation in this function is derived
        // from the "Shoelace Formula" (see: https://en.wikipedia.org/wiki/Shoelace_formula).
        dest_vals[ei] = 0.0;
        for(index_t eci = 0; eci < elem_size; eci++)
        {
            float64 coord_vals[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
            for(index_t cdi = 0; cdi < 2; cdi++)
            {
                index_t ci = ((eci + cdi) % elem_size) + elem_start_index;
                data_node.set_external(conn_dtype, (void*)topo_conn.element_ptr(ci));
                index_t icoord = data_node.to_int64();

                float64 *cvals = &coord_vals[cdi][0];
                for(index_t di = 0; di < 2; di++)
                {
                    const Node &axis_coords = coords["values"][elem_axes[di]];
                    data_node.set_external(DataType(axis_coords.dtype().id(), 1),
                        (void*)axis_coords.element_ptr(icoord));
                    cvals[di] = data_node.to_float64();
                }
            }

            dest_vals[ei] += coord_vals[0][0] * coord_vals[1][1];
            dest_vals[ei] -= coord_vals[0][1] * coord_vals[1][0];
        }
        dest_vals[ei] = std::abs(dest_vals[ei] / 2.0);
    }
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
        const float64 mesh_sides_volume =
            calc_mesh_elem_volume(ti, &MPDIMS[0]) / sides_per_elem;

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

            { // Validate Correctness of Element Integrity //
                Node side_invs;
                calc_inversion_field(ti, side_topo, side_coords, side_invs);
                {
                    Node side_inv_int64;
                    side_invs["values"].to_int64_array(side_inv_int64);
                    int64_array side_inv_data = side_inv_int64.as_int64_array();

                    std::vector<int64> side_inv_expected_vector(mesh_sides, 0);
                    int64_array side_inv_expected(&side_inv_expected_vector[0],
                        DataType::int64(mesh_sides));

                    EXPECT_FALSE(side_inv_data.diff(side_inv_expected, info));
                }

                if(!is_mesh_3d)
                {
                    Node side_vols;
                    calc_volume_field(ti, side_topo, side_coords, side_vols);
                    {
                        Node side_vol_float64;
                        side_vols["values"].to_float64_array(side_vol_float64);
                        float64_array side_vol_data = side_vol_float64.as_float64_array();

                        std::vector<float64> side_vol_expected_vector(mesh_sides, mesh_sides_volume);
                        float64_array side_vol_expected(&side_vol_expected_vector[0],
                            DataType::float64(mesh_sides));

                        EXPECT_FALSE(side_vol_data.diff(side_vol_expected, info));
                    }
                }
            }

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
        const float64 mesh_corners_volume =
            calc_mesh_elem_volume(ti, &MPDIMS[0]) / corners_per_elem;

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

            { // Validate Correctness of Element Integrity //
                Node corner_invs;
                calc_inversion_field(ti, corner_topo, corner_coords, corner_invs);
                {
                    Node corner_inv_int64;
                    corner_invs["values"].to_int64_array(corner_inv_int64);
                    int64_array corner_inv_data = corner_inv_int64.as_int64_array();

                    std::vector<int64> corner_inv_expected_vector(mesh_corners, 0);
                    int64_array corner_inv_expected(&corner_inv_expected_vector[0],
                        DataType::int64(mesh_corners));

                    EXPECT_FALSE(corner_inv_data.diff(corner_inv_expected, info));
                }

                if(!is_mesh_3d)
                {
                    Node corner_vols;
                    calc_volume_field(ti, corner_topo, corner_coords, corner_vols);
                    {
                        Node corner_vol_float64;
                        corner_vols["values"].to_float64_array(corner_vol_float64);
                        float64_array corner_vol_data = corner_vol_float64.as_float64_array();

                        std::vector<float64> corner_vol_expected_vector(mesh_corners, mesh_corners_volume);
                        float64_array corner_vol_expected(&corner_vol_expected_vector[0],
                            DataType::float64(mesh_corners));

                        EXPECT_FALSE(corner_vol_data.diff(corner_vol_expected, info));
                    }
                }
            }

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
